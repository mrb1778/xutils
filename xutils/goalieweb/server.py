from datetime import datetime
from blacksheep import Application, json

from xutils.goalie.goal import goal, InputParam

app = Application()


@app.route("/")
def home():
    return f"Hello, World! {datetime.utcnow().isoformat()}"


@app.route("/list-goals")
def list_goals():
    return json(goal.goal_names())
