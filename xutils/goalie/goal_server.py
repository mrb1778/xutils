import importlib
from typing import Union

from fastapi import FastAPI, HTTPException
from starlette.responses import Response

from xutils.goalie import goal
from xutils.goalie.goal import UnmetGoal
import xutils.data.json_utils as ju

app = FastAPI()


def json_response(obj) -> Response:
    return Response(content=ju.write(obj), media_type='application/json')


@app.get("/")
def read_root():
    return {"status": "Up"}


@app.get("/goals/scope/list")
def list_scopes():
    return json_response(goal.lib.keys())


@app.get("/goals/all/list")
def list_all_goals():
    return json_response(goal.all_goals())


@app.get("/goals/list/{scope}")
def list_goals(scope):
    return json_response(goal.goal_names(scope=scope))


@app.get("/goals/list")
def list_goals_no_scope():
    return list_goals(scope=None)


@app.get("/goals/load/{path}")
def load_goals(path):
    with goal.in_scope(path):
        importlib.import_module(path)
    return "loaded"


@app.get("/goals/inputs/{scope}/{goal_name}")
def goal_inputs(scope, goal_name):
    return json_response(goal.inputs(scope=scope, goal=goal_name))


@app.get("/goals/inputs/{goal_name}")
def goal_inputs_no_scope(goal_name):
    return goal_inputs(goal_name=goal_name, scope=None)


@app.get("/goals/inputs/required/{scope}/{goal_name}")
def goal_required_inputs(scope, goal_name):
    return json_response(goal.inputs(scope=scope, goal=goal_name, only_required=True))


@app.get("/goals/inputs/required/{goal_name}")
def goal_required_inputs_no_scope(goal_name):
    return goal_required_inputs(goal_name=goal_name, scope=None)


@app.post("/goals/run/{scope}/{goal_name}")
def run_goal(scope, goal_name, params: dict):
    try:
        return json_response(goal.run(goal_name, scope=scope, **params))
    except UnmetGoal as ug:
        raise HTTPException(status_code=500, detail=ug.message)


@app.post("/goals/run/{goal_name}")
def run_goal_no_scope(goal_name, params: dict):
    return run_goal(scope=None, goal_name=goal_name, params=params)
