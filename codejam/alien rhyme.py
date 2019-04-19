t = int(input()) # read a line with a single integer
for i in range(1, t + 1):
  number = int(input()) # read integer
  words = []
  selected_words = []
  word_subset_count = 0
  for j in range(number):
      word = input()
      words.append(word)

  # print(words)
  for m in range(len(words)):
      word = words[m]
      #assign a accent letter
      the_other_words = []
      if word not in selected_words:
          for k in range(len(word)):
              acc_letter = word[k]
              acc_suffix = word[k+1:]
              # print("Try:",k)
              # print(acc_letter,acc_suffix)
              #find this suffix in other words
              matchs = 0
              word_list = list(set(words) - set(selected_words))
              # print(word_list)
              for w in word_list:
                  if w != word:
                      if len(acc_suffix) != 0:
                          pos = str(w).find(str(acc_suffix))
                          if pos != -1 and pos != 0:
                              # print("FOUND AT pos:", pos)
                              #suffix found check rest of the length
                              if len(acc_suffix) == len(w[pos:]):
                                  #also check accented letter
                                  if acc_letter ==  w[pos-1]:
                                      #match found
                                      # print("match found")
                                      matchs +=1
                                      the_other_words.append(w)
                      else:
                          #just check the accent letter
                          if acc_letter == w[-1]:
                              # match found
                              # print("match found else")
                              matchs += 1
                              the_other_words.append(w)
              if matchs == 1 or matchs>1:
                  #we have a perfect match
                  # print("perfect match found")
                  word_subset_count +=2
                  #remove those 2 words
                  selected_words.append(word)
                  selected_words.extend(the_other_words)
                  # print("selected words:",selected_words)
                  break
  print("Case #{}: {}".format(i, word_subset_count))