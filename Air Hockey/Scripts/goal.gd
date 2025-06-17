class_name Goal

extends Node

signal goal(goalside)

enum Side {LEFT, RIGHT}

@export var Goal_Side: Side

func _on_body_entered(body):
	if body is Pack:
		goal.emit(Goal_Side)
