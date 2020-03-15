#!/bin/bash
# argument-1 => type of collaborative learning training: "key-protected" or "vanilla"
# argument-2 => dataset
# argument-3 => (optional for key-protected classification) 0 or 1 to denote whether to use fixed layer or not

if [[ "$#" < 2 ]]; then
    echo "Some arguments are missing. Check the header of the file."
    exit -1
fi

OUTPUT_ROOT="/data/privacy"

train_vanilla_coll () {
    # argument-1 => dataset name
    echo "Vanilla collaborative learning experiment, without any attacker"

    output_dir="${OUTPUT_ROOT}/${1}/vanilla_coll"
    rm -rf ${output_dir}
    mkdir -p ${output_dir} 

    # overwritten some of the default hyper-parameters
    if [[ "${1}" == "mnist" ]]; then
        batch_size=128
    elif [[ "${1}" == "olivetti" ]]; then
        batch_size=50
    fi

    python train_coll.py \
        --dataset=$1 \
        --output_dir=${output_dir} \
        --n_epochs=100 \
        --batch_size=${batch_size} | tee ${output_dir}/program.txt
}

train_keyprot_coll () {
    # argument-1 => dataset name
    # argument-2 => key dimension
    # argument-3 => whether to use a fixed layer
    echo "********************************************************************************"
    echo "Key-protected collaborative learning, without any attacker"
    echo "dataset: ${1}"
    echo "d_key: ${2}"
    echo "fixed_layer: ${3}"

    output_dir="${OUTPUT_ROOT}/${1}/keyprot_coll/dkey-${2}_fl-${3}"
    rm -rf ${output_dir}
    mkdir -p ${output_dir} 

    # overwritten some of the default hyper-parameters
    if [[ "${1}" == "mnist" ]]; then
        batch_size=128
    elif [[ "${1}" == "olivetti" ]]; then
        batch_size=50
    fi

    python train_coll.py \
        --dataset=$1 \
        --output_dir=${output_dir} \
        --key_prot \
        --d_key $2 \
        --fixed_layer $3 \
        --n_epochs=100 \
        --batch_size=${batch_size} | tee ${output_dir}/program.txt
}


if [[ "${1}" == "vanilla" ]]; then
    train_vanilla_coll ${2}

elif [[ "${1}" == "key-protected" ]]; then
    train_keyprot_coll ${2} 128 ${3}
    train_keyprot_coll ${2} 1024 ${3}
    train_keyprot_coll ${2} 4096 ${3}
    train_keyprot_coll ${2} 16384 ${3}
fi
