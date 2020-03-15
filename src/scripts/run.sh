#!/bin/bash

##############################
# collaborative learning experiment WITHOUT an attacker
##############################

# argument-1 => type of collaborative learning training: "key-protected" or "vanilla"
# argument-2 => dataset
# argument-3 => (optional for key-protected classification) 0 or 1 to denote whether to use fixed layer or not
# bash scripts/clf_training.sh "key-protected" "mnist" 0
# bash scripts/clf_training.sh "key-protected" "mnist" 1
# bash scripts/clf_training.sh "key-protected" "olivetti" 0
# bash scripts/clf_training.sh "key-protected" "olivetti" 1


##############################
# collaborative learning experiment WITH an attacker having the same class key
##############################

# argument-1 => type of collaborative learning training: "key-protected" or "vanilla"
# argument-2 => dataset
# argument-3 => class id to attack
# argument-4 => (optional for key-protected classification) 0 or 1 to denote whether to use fixed layer or not
# argument-5 => (optional for key-protected classification) string to denote epsilon

# bash ./scripts/gan_attack.sh "key-protected" "mnist" 0 0 "zero"
# bash ./scripts/gan_attack.sh "key-protected" "mnist" 0 1 "zero"
# bash ./scripts/gan_attack.sh "key-protected" "mnist" 1 0 "zero"
# bash ./scripts/gan_attack.sh "key-protected" "mnist" 1 1 "zero"
# bash ./scripts/gan_attack.sh "key-protected" "mnist" 2 0 "zero"
# bash ./scripts/gan_attack.sh "key-protected" "mnist" 2 1 "zero"
# bash ./scripts/gan_attack.sh "key-protected" "mnist" 3 0 "zero"
# bash ./scripts/gan_attack.sh "key-protected" "mnist" 3 1 "zero"
# bash ./scripts/gan_attack.sh "key-protected" "mnist" 4 0 "zero"
# bash ./scripts/gan_attack.sh "key-protected" "mnist" 4 1 "zero"

##############################
# collaborative learning experiment WITH an attacker having a random class key
# note that for this setup, the value of c_attack does not matter
##############################

bash ./scripts/gan_attack.sh "key-protected" "mnist" 0 0 "random"
bash ./scripts/gan_attack.sh "key-protected" "mnist" 0 1 "random"
