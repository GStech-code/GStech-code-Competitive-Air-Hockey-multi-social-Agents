extends Node

var L_score := 0
var R_score := 0

@onready var left_score: Label = %Left_Score
@onready var right_score: Label = %Right_Score


func add_point_left():
	L_score += 1
	left_score.text = str(L_score)

func add_point_right():
	R_score += 1
	right_score.text = str(R_score)
