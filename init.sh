#!/usr/bin/env bash

mkdir -p tf/proto
protoc --proto_path=libs/lczero-common --python_out=tf libs/lczero-common/proto/net.proto
protoc --proto_path=libs/lczero-common --python_out=tf libs/lczero-common/proto/chunk.proto
touch tf/proto/__init__.py
