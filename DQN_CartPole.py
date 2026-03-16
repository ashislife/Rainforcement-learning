import random

# Initial values
cart_position = 0
pole_angle = 0
steps = 0

print("CartPole Simulation Started")

while True:
    
    # choose random action
    action = random.choice(["left", "right"])
    
    # update cart position
    if action == "left":
        cart_position -= 1
        pole_angle += 1
    else:
        cart_position += 1
        pole_angle -= 1

    steps += 1

    print("Step:", steps,
          "Action:", action,
          "Cart Position:", cart_position,
          "Pole Angle:", pole_angle)

    # check if pole falls
    if abs(pole_angle) > 10:
        print("Pole Fell! Game Over")
        break

print("Total Score:", steps)