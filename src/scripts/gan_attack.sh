#!/bin/bash
# argument-1 => type of collaborative learning training: "key-protected" or "vanilla"
# argument-2 => dataset
# argument-3 => class id to attack
# argument-4 => (optional for key-protected classification) 0 or 1 to denote whether to use fixed layer or not
# argument-5 => (optional for key-protected classification) string to denote epsilon

if [[ "$#" < 3 ]]; then
    echo "Some arguments are missing. Check the header of the file."
    exit -1
fi

# root directory where to store all the output
OUTPUT_ROOT="/data/privacy"
# OUTPUT_ROOT="/tmp-network/user/mbsariyi/privacy"
# OUTPUT_ROOT="/local/mbsariyi/privacy"

train_vanilla_coll () {
    # args:
    # argument-1 => dataset
    # argument-2 => c_attack
    echo "********************************************************************************"
    echo "Vanilla collaborative learning experiment"
    echo "c_attack: ${2}"

    # overwritten some of the default hyper-parameters
    if [[ "${1}" == "mnist" ]]; then
        n_steps_train_gen=150
        n_fake_to_trainset=2000
        batch_size=128
        n_epochs=100
    elif [[ "${1}" == "olivetti" ]]; then
        n_steps_train_gen=50
        n_fake_to_trainset=10
        n_previous_to_trainset=5
        batch_size=50
        n_epochs=1000
    fi

    output_dir="${OUTPUT_ROOT}/${1}/gan_attack/${2}"
    rm -rf ${output_dir}
    mkdir -p ${output_dir}
    
    python train_coll.py \
        --dataset=${1} \
        --c_attack=${2} \
        --output_dir=${output_dir} \
        --n_epochs=${n_epochs} \
        --batch_size=${batch_size} \
        --n_steps_train_gen=${n_steps_train_gen} \
        --n_previous_to_trainset=${n_previous_to_trainset} \
        --n_fake_to_trainset=${n_fake_to_trainset} | tee ${output_dir}/program.txt
}

train_keyprot_coll () {
    # args:
    # argument-1 => dataset
    # argument-2 => c_attack
    # argument-3 => key dimension
    # argument-4 => whether to use a fixed layer
    # argument-5 => epsilon
    echo "********************************************************************************"
    echo "Key-protected collaborative learning, with attacker"
    echo "dataset: ${1}"
    echo "c_attack: ${2}"
    echo "d_key: ${3}"
    echo "fixed_layer: ${4}"
    echo "epsilon: ${4}"

    # overwritten some of the default hyper-parameters
    if [[ "${1}" == "mnist" ]]; then
        n_steps_train_gen=200
        n_fake_to_trainset=2000
        n_previous_to_trainset=5
        batch_size=128
        n_epochs=250
    elif [[ "${1}" == "olivetti" ]]; then
        n_steps_train_gen=50
        n_fake_to_trainset=10
        n_previous_to_trainset=5
        batch_size=50
        n_epochs=1000
    fi

    output_dir="${OUTPUT_ROOT}/${1}/keyprot_gan-attack/catt-${2}__dkey-${3}__fl-${4}__eps-${5}"
    rm -rf ${output_dir}
    mkdir -p ${output_dir}
    
    python train_coll.py \
        --dataset=${1} \
        --key_prot \
        --c_attack=${2} \
        --d_key ${3} \
        --fixed_layer ${4} \
        --epsilon ${5} \
        --output_dir=${output_dir} \
        --n_epochs=${n_epochs} \
        --batch_size=${batch_size} \
        --n_steps_train_gen=${n_steps_train_gen} \
        --n_previous_to_trainset=${n_previous_to_trainset} \
        --n_fake_to_trainset=${n_fake_to_trainset} | tee ${output_dir}/program.txt
}


# for mnist, c_attack's = 0, 1, 2, 3, 4
# for olivetti, c_attack's = 1, 13, 21, 24, 25
if [[ "${1}" == "vanilla" ]]; then
    train_vanilla_coll ${2}

elif [[ "${1}" == "key-protected" ]]; then
    train_keyprot_coll ${2} ${3} 128 ${4} ${5}
    train_keyprot_coll ${2} ${3} 1024 ${4} ${5}
    train_keyprot_coll ${2} ${3} 4096 ${4} ${5}
    train_keyprot_coll ${2} ${3} 16384 ${4} ${5}
fi



