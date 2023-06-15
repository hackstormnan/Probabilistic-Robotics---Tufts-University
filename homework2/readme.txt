For the normal part, it can be runned directly. 
For the extra credit, I commented them. You can uncomment the extra credit code and comment the related normal part of code. 
For extra credit 3, there is no normal code that need to be commented


For example:
this is the normal code:
    # add movement to the particles (normal)
    # subsititure part starts

    for i in particles:
        i[0] += movement_x
        i[1] += movement_y
        
    # subsititure part ends

this is the extra credit part:
    # # add movement to the particles (add noise, extra credit 1)
    # subsititure part starts

    # for i in particles:
    #     i[0] += movement_x + np.random.normal(0, 10)
    #     i[1] += movement_y + np.random.normal(0, 10)

    # subsititure part ends

and just comment the normal part while uncommenting the extra credit part



Hit the continue at the command bar to let drone and particles move