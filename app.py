from flask import Flask, render_template, request, redirect, session, url_for
import csv
import os

app = Flask(__name__)

app.secret_key = "change_this_secret_key"

ATTENDANCE_FILE = "attendance.csv"

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"


def read_attendance():
    records = []

    if not os.path.exists(ATTENDANCE_FILE):
        return records

    with open(ATTENDANCE_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    return records


@app.route("/login", methods=["GET", "POST"])
def login():
    error = ""

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("attendance"))
        else:
            error = "Invalid username or password"

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
def attendance():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    records = read_attendance()

    search_name = request.args.get("name", "").lower()
    filter_date = request.args.get("date", "")

    if search_name:
        records = [
            r for r in records
            if search_name in r["name"].lower()
        ]

    if filter_date:
        records = [
            r for r in records
            if r["date"] == filter_date
        ]

    records = sorted(records, key=lambda x: (x["date"], x["time"]), reverse=True)

    return render_template(
        "attendance.html",
        records=records,
        search_name=search_name,
        filter_date=filter_date
    )


@app.route("/update", methods=["POST"])
def update():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    name = request.form["name"]
    date = request.form["date"]
    time = request.form["time"]
    new_status = request.form["status"]

    rows = []

    with open(ATTENDANCE_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (
                row["name"] == name and
                row["date"] == date and
                row["time"] == time
            ):
                row["status"] = new_status
            rows.append(row)

    with open(ATTENDANCE_FILE, "w", newline="") as f:
        fieldnames = ["name", "date", "time", "status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return redirect(url_for("attendance"))


if __name__ == "__main__":
    app.run(debug=True)