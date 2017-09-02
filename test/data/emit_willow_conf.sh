#!/bin/bash
set -e -u

if (( "$#" != 1 )); then
    echo "$0 WILLOW_DATABASE_PATH" >&2
    exit 1
fi

WILLOW_DB_PATH="$1"

INDENTATION=""
function indent {
    INDENTATION="  $INDENTATION"
}

function unindent {
    INDENTATION="${INDENTATION#  }"
}

function emit {
    echo "${INDENTATION}$1"
}

function emit_array_elem {
    echo "${INDENTATION}- $1"
}

function emit_model {
    emit 'models:'
    shopt -s nullglob
    for object in ${WILLOW_DB_PATH}/*models/*; do
        if [[ -d $object ]]; then
            indent
            emit "$(basename "$object"):"
            if [[ -f "${object}/3D_model.pcd" ]]; then
                indent
                emit "full: ${object}/3D_model.pcd"
                unindent
            fi
            if [[ -d "${object}/views" ]]; then
                indent
                emit "views:"
                for cloud in ${object}/views/cloud_*.pcd; do
                    indent
                    cloud_basename="$(basename "$cloud" .pcd)"
                    view_idx="${cloud_basename#cloud_}"
                    emit "- "
                    indent
                    emit "cloud: $cloud"
                    emit "indices: ${object}/views/object_indices_${view_idx}.txt"
                    emit "pose: ${object}/views/pose_${view_idx}.txt"
                    unindent
                    unindent
                done
                unindent
            fi
            unindent
        fi
    done
}

function emit_scenes {
    emit 'scenes:'
    shopt -s nullglob
    for scene in ${WILLOW_DB_PATH}/willow_test_set/*; do
        if [[ -d $scene ]]; then
            indent
            emit "$(basename "$scene"):"
            for cloud in ${scene}/cloud_*.pcd; do
                indent
                emit_array_elem ""
                indent
                emit "cloud: $cloud"
                emit_instances "$(basename "$scene")" "$(basename "$cloud" '.pcd')"
                unindent
                unindent
            done
            unindent
        fi
    done
}

function emit_instances {
    scene="$1"
    cloud="$2"
    emit "instances:"
    indent
    instance_prefix="${WILLOW_DB_PATH}/willow_annotations/willow/${scene}/${cloud}_"
    shopt -s nullglob
    for pose in "${instance_prefix}"object_*.txt; do
        emit_array_elem ""
        indent
        instance="${pose#$instance_prefix}"
        object="${instance%_*.txt}"
        emit "object: $object"
        emit "pose: $pose"
        occlusion="${instance_prefix}occlusion_${instance}"
        if [[ -f $occlusion ]]; then
            emit "occlusion: $occlusion"
        fi
        unindent
    done
    unindent
}

function emit_willow {
    emit_model
    emit_scenes
}

emit_willow