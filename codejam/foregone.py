"""
Someone just won the Code Jam lottery, and we owe them N jamcoins!
However, when we tried to print out an oversized check, we encountered a problem.
The value of N, which is an integer, includes at least one digit that is a 4...
and the 4 key on the keyboard of our oversized check printer is broken.

Fortunately, we have a workaround: we will send our winner two checks for positive integer amounts A and B,
such that neither A nor B contains any digit that is a 4, and A + B = N.
Please help us find any pair of values A and B that satisfy these conditions.

Input

Output

3
4
940
4444


Case #1: 2 2
Case #2: 852 88
Case #3: 667 3777
"""


t = int(input()) # read a line with a single integer
for i in range(1, t + 1):
  number = int(input()) # read integer
  a=b=c=number
  k = True
  while k:
      #find where is 4
      pos = len(str(c)) - str(c).find('4')
      sub = 10**(pos-1)
      a = int(c - sub)
      c = a
      if '4' in str(a):
          continue
      else:
          b = int(number - a)
          break
  print("Case #{}: {} {}".format(i, a, b))