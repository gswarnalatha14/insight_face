from flask import (
    Flask, render_template, request, redirect,
    url_for, session, jsonify
)
from pymongo import MongoClient
from werkzeug.security import check_password_hash, generate_password_hash
from bson.objectid import ObjectId
from functools import wraps
from datetime import datetime

import cv2
import numpy as np

# 🔥 InsightFace engine
from face_engine import recognize_faces

# =========================
# APP
# =========================
app = Flask(__name__)
app.secret_key = "attendai_secret"

# =========================
# DATABASE
# =========================
client = MongoClient("mongodb://localhost:27017/")
db = client["attendance_db"]

teachers_col = db["Teachers"]
classes_col = db["Classes"]
attendance_col = db["Markattendance"]
timetable_col = db["timetable"]
students_col = db["Students"]

# =========================
# SESSION FACE CACHE
# =========================
CURRENT_SESSION_PRESENT = set()

# =========================
# TIMETABLE CONFIG
# =========================
TIME_SLOTS = [
    "09:00-09:50","09:50-10:40","10:40-11:00",
    "11:00-11:50","11:50-12:40","12:40-13:50",
    "13:50-14:40","14:40-15:30","15:30-16:20"
]

DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]

# =========================
# AUTH DECORATOR
# =========================
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if "teacher_id" not in session:
            return redirect(url_for("home"))
        return f(*args, **kwargs)
    return wrap

# =========================
# ACTIVE TIMETABLE CHECK
# =========================
def get_active_timetable(subject, department):
    now = datetime.now()
    return timetable_col.find_one({
        "day": now.strftime("%A"),
        "subject": subject,
        "department": department,
        "start": {"$lte": now.strftime("%H:%M")},
        "end": {"$gte": now.strftime("%H:%M")}
    })

# =========================
# LOGIN
# =========================
@app.route("/")
def home():
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def login():
    teacher = teachers_col.find_one({"username": request.form["username"]})

    if not teacher or not check_password_hash(
        teacher["password"], request.form["password"]
    ):
        return render_template("login.html", error="Invalid credentials")

    session["teacher_id"] = teacher["teacher_id"]
    session["teacher_name"] = teacher["name"]

    return redirect(url_for("dashboard"))

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# =========================
# CHANGE PASSWORD
# =========================
@app.route("/change_password", methods=["GET","POST"])
@login_required
def change_password():

    if request.method == "POST":
        teacher = teachers_col.find_one(
            {"teacher_id": session["teacher_id"]}
        )

        if not check_password_hash(
            teacher["password"], request.form["old_password"]
        ):
            return render_template(
                "change_password.html",
                error="Old password incorrect"
            )

        teachers_col.update_one(
            {"teacher_id": session["teacher_id"]},
            {"$set":{
                "password": generate_password_hash(
                    request.form["new_password"]
                )
            }}
        )

        return render_template(
            "change_password.html",
            success="Password updated"
        )

    return render_template("change_password.html")

# =========================
# DASHBOARD
# =========================
@app.route("/dashboard")
@login_required
def dashboard():

    classes = list(classes_col.find())
    enriched = []

    for cls in classes:
        tt = timetable_col.find_one({
            "subject": cls["subject"],
            "department": cls["class_name"]
        })

        enriched.append({
            "_id": cls["_id"],
            "subject": cls["subject"],
            "class_name": cls["class_name"],
            "teacher_name": cls["teacher_name"],
            "room": tt["room"] if tt else "-",
            "start": tt["start"] if tt else "",
            "end": tt["end"] if tt else ""
        })

    return render_template(
        "dashboard.html",
        teacher=session["teacher_name"],
        classes=enriched
    )

# =========================
# LIVE ATTENDANCE PAGE
# =========================
@app.route("/live_attendance/<class_id>")
@login_required
def live_attendance(class_id):

    class_doc = classes_col.find_one({"_id": ObjectId(class_id)})
    class_doc["_id_str"] = str(class_doc["_id"])

    session["last_subject"] = class_doc["subject"]
    session["last_department"] = class_doc["class_name"]

    return render_template(
        "mark_attendance.html",
        class_doc=class_doc
    )

# =========================
# PROCESS FRAME (InsightFace)
# =========================
@app.route("/process_frame", methods=["POST"])
@login_required
def process_frame():

    subject = session.get("last_subject")
    department = session.get("last_department")

    if not get_active_timetable(subject, department):
        return jsonify({"error":"Class not active"}), 403

    frame = cv2.imdecode(
        np.frombuffer(request.data, np.uint8),
        cv2.IMREAD_COLOR
    )

    detections = recognize_faces(frame)

    new_names = []

    for det in detections:
        name = det["name"]

        if name == "Unknown":
            continue

        if name in CURRENT_SESSION_PRESENT:
            continue

        CURRENT_SESSION_PRESENT.add(name)
        new_names.append(name)

        print("Detected:", name)

    return jsonify({"new": new_names}), 200

# =========================
# STOP → CONFIRM
# =========================
@app.route("/stop_attendance", methods=["POST"])
@login_required
def stop_attendance():
    return redirect(url_for("confirm_attendance"))

# =========================
# CONFIRM PAGE
# =========================
@app.route("/confirm_attendance")
@login_required
def confirm_attendance():

    return render_template(
        "confirm_attendance.html",
        students=sorted(list(CURRENT_SESSION_PRESENT)),
        subject=session.get("last_subject")
    )

# =========================
# FINALIZE
# =========================
@app.route("/finalize_attendance", methods=["POST"])
@login_required
def finalize_attendance():

    selected = request.form.getlist("students")
    subject = session.get("last_subject")
    department = session.get("last_department")
    today = datetime.now().strftime("%Y-%m-%d")

    active_class = get_active_timetable(subject, department)

    if not active_class:
        CURRENT_SESSION_PRESENT.clear()
        return redirect(url_for("dashboard"))

    for name in selected:

        student = students_col.find_one({
            "name":{"$regex":f"^{name}$","$options":"i"}
        })

        if not student:
            continue

        if attendance_col.find_one({
            "student_id": student["student_id"],
            "subject": subject,
            "date": today
        }):
            continue

        attendance_col.insert_one({
            "student_id": student["student_id"],
            "student_name": student["name"],
            "subject": subject,
            "date": today,
            "day": datetime.now().strftime("%A"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "start_time": active_class["start"],
            "end_time": active_class["end"],
            "status": "Present"
        })

    CURRENT_SESSION_PRESENT.clear()
    return redirect(url_for("dashboard"))

# =========================
# SUMMARY
# =========================
@app.route("/attendance_summary")
@login_required
def attendance_summary():

    subject = session.get("last_subject")
    today = datetime.now().strftime("%Y-%m-%d")

    present_students = list(attendance_col.find(
        {"subject":subject,"date":today},
        {"_id":0,"student_id":1,"student_name":1}
    ))

    return render_template(
        "attendance_summary.html",
        subject=subject,
        present_students=present_students
    )

# =========================
# STUDENTS REPORT
# =========================
@app.route("/students")
@login_required
def students():

    TOTAL_CLASSES = 60

    pipeline = [
        {"$group":{
            "_id":{"student_id":"$student_id","subject":"$subject"},
            "name":{"$first":"$student_name"},
            "attended":{"$sum":1}
        }},
        {"$group":{
            "_id":"$_id.student_id",
            "name":{"$first":"$name"},
            "subjects":{"$push":{
                "subject":"$_id.subject",
                "attended":"$attended"
            }},
            "total_attended":{"$sum":"$attended"}
        }},
        {"$project":{
            "_id":0,
            "student_id":"$_id",
            "name":1,
            "subjects":1,
            "percentage":{
                "$round":[
                    {"$multiply":[
                        {"$divide":["$total_attended",TOTAL_CLASSES]},
                        100
                    ]},2]
            }
        }}
    ]

    students = list(attendance_col.aggregate(pipeline))
    return render_template("students.html", students=students)

# =========================
# TEACHERS REPORT
# =========================
@app.route("/teachers")
@login_required
def teachers():

    teachers = list(teachers_col.find())
    attendance = list(attendance_col.find())

    teacher_data = []

    for t in teachers:
        subjects = [
            c["subject"]
            for c in classes_col.find(
                {"teacher_name":t["name"]}
            )
        ]

        uniq = set()

        for a in attendance:
            if a["subject"] in subjects:
                uniq.add((a["subject"], a["date"]))

        teacher_data.append({
            "name": t["name"],
            "section": t.get("department","-"),
            "classes_taken": len(uniq)
        })

    return render_template("teachers.html", teachers=teacher_data)

# =========================
# TIMETABLE VIEW
# =========================
@app.route("/timetable")
@login_required
def timetable():

    data = list(timetable_col.find({},{"_id":0}))
    lookup = {day:{} for day in DAYS}

    for d in data:
        key = f"{d['start']}-{d['end']}"
        lookup[d["day"]][key] = d["subject"]

    grid = {}

    for day in DAYS:
        row = []
        for slot in TIME_SLOTS:
            subject = lookup[day].get(slot,"")
            row.append({
                "type":"class" if subject else "empty",
                "label":subject,
                "colspan":1
            })
        grid[day] = row

    return render_template(
        "timetable.html",
        time_slots=TIME_SLOTS,
        grid=grid
    )

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)
