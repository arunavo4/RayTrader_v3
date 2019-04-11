"""
On the Code Jam team, we enjoy sending each other pangrams, which are phrases that use each letter of the English alphabet at least once. One common example of a pangram is "the quick brown fox jumps over the lazy dog". Sometimes our pangrams contain confidential information — for example, CJ QUIZ: KNOW BEVY OF DP FLUX ALGORITHMS — so we need to keep them secure.

We looked through a cryptography textbook for a few minutes, and we learned that it is very hard to factor products of two large prime numbers, so we devised an encryption scheme based on that fact. First, we made some preparations:

We chose 26 different prime numbers, none of which is larger than some integer N.
We sorted those primes in increasing order. Then, we assigned the smallest prime to the letter A, the second smallest prime to the letter B, and so on.
Everyone on the team memorized this list.
Now, whenever we want to send a pangram as a message, we first remove all spacing to form a plaintext message. Then we write down the product of the prime for the first letter of the plaintext and the prime for the second letter of the plaintext. Then we write down the product of the primes for the second and third plaintext letters, and so on, ending with the product of the primes for the next-to-last and last plaintext letters. This new list of values is our ciphertext. The number of values is one smaller than the number of characters in the plaintext message.

For example, suppose that N = 103 and we chose to use the first 26 odd prime numbers, because we worry that it is too easy to factor even numbers. Then A = 3, B = 5, C = 7, D = 11, and so on, up to Z = 103. Also suppose that we want to encrypt the CJ QUIZ... pangram above, so our plaintext is CJQUIZKNOWBEVYOFDPFLUXALGORITHMS. Then the first value in our ciphertext is 7 (the prime for C) times 31 (the prime for J) = 217; the next value is 1891, and so on, ending with 3053.

We will give you a ciphertext message and the value of N that we used. We will not tell you which primes we used, or how to decrypt the ciphertext. Do you think you can recover the plaintext anyway?

Input

2
103 31
217 1891 4819 2291 2987 3811 1739 2491 4717 445 65 1079 8383 5353 901 187 649 1003 697 3239 7663 291 123 779 1007 3551 1943 2117 1679 989 3053
10000 25
3292937 175597 18779 50429 375469 1651121 2102 3722 2376497 611683 489059 2328901 3150061 829981 421301 76409 38477 291931 730241 959821 1664197 3057407 4267589 4729181 5335543


Output

Case #1: CJQUIZKNOWBEVYOFDPFLUXALGORITHMS
Case #2: SUBDERMATOGLYPHICFJKNQVWXZ
"""

import math

def primeFactors(n,l):
    prime_list = []
    # Print the number of two's that divide n
    while n % 2 == 0:
        if l==25:
            prime_list.append(2)
        n = n / 2
    # n must be odd at this point
    # so a skip of 2 ( i = i + 2) can be used
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        # while i divides n , print i ad divide n
        while n % i == 0:
            if i%2!=0:
                prime_list.append(int(i))
            n = n / i
            # Condition if n is a prime
    # number greater than 2
    if n > 2:
        if n % 2 != 0:
            prime_list.append(int(n))

    return prime_list

t = int(input()) # read a line with a single integer
for i in range(1, t + 1):
  n, m = [int(s) for s in input().split(" ")] # read a list of integers, 2 in this case
  cipher_text = [int(s) for s in input().split(" ")] # read the cipher text
  last_prime_set = []
  code_origin = []
  chars_list = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
  for code in cipher_text:
      primes = primeFactors(code,m)
      if len(code_origin)!=0:
          for prime in last_prime_set:
              if prime in primes:
                  primes.remove(prime)
          code_origin.extend(primes)
          last_prime_set = primes
      else:
          last_prime_set = primes
          code_origin.extend(primes)
      print("Number:{} primes: {}".format(code,primes))

  #remove duplicates
  code_sorted = list(set(code_origin.copy()))
  #sort
  code_sorted.sort()
  #TURN INTO A DICT
  alpha_dict = dict(zip(code_sorted,chars_list))
  message = ""
  for code in code_origin:
      message += alpha_dict[code]
  print("Primes set len {} : {}".format(len(code_origin), code_origin))
  print("Primes set_sorted len {} : {}".format(len(code_sorted), code_sorted))
  print("Dict: ",alpha_dict)
  print("Case #{}: {}".format(i, message))

# primeFactors(n)