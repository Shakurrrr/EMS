from flask import render_template, redirect, url_for

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/add_user_form", methods=["GET", "POST"])
def add_user_form():
    if request.method == "POST":
        username = request.form.get("username")
        if not username:
            return render_template("add_user.html", error="Username is required")
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return render_template("add_user.html", error="Username already exists")
        new_user = User(username=username)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for("get_users_ui"))
    return render_template("add_user.html")

@app.route("/users_ui")
def get_users_ui():
    users = User.query.all()
    return render_template("users.html", users=users)
