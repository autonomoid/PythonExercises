import math
for x in range(1,101):
    if x % 3 == 0:
        if x % 5 == 0:
            print("Fizzbuzz")
        else:
            print("Fizz")
    elif x % 5 == 0:
        print("Buzz")
    else:
        print(x)