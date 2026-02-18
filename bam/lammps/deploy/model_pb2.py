# -*- coding: utf-8 -*-
# Generated protocol buffer code for chemtrain-deploy model format.
# Compatible with protobuf >= 3.20.0
# Original proto: chemtrain-deploy/connector/model.proto
#
# Copyright 2025 Multiscale Modeling of Fluid Materials, TU Munich
# Licensed under the Apache License, Version 2.0

"""Protobuf definitions for chemtrain-deploy model serialization."""

from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

_sym_db = _symbol_database.Default()

# Serialized FileDescriptor for model.proto (version-independent binary)
# This encodes the proto schema:
#   message Model {
#     bytes mlir_module = 1;
#     enum NeighborListType { UNSPECIFIED=0; DEVICE_SPARSE=1; SIMPLE_DENSE=2; SIMPLE_SPARSE=3; }
#     message NeighborList {
#       NeighborListType type = 1; float cutoff = 2; repeated int32 nbr_order = 3;
#       optional bool half_list = 4; repeated string statistics_keys = 5;
#     }
#     NeighborList neighbor_list = 4;
#     optional string unit_style = 5;
#     repeated string quantities = 6;
#   }
DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x0bmodel.proto\x12\x03jcn\"\x83\x03\n\x05Model\x12\x13\n\x0bmlir_module'
    b'\x18\x01 \x01(\x0c\x12.\n\rneighbor_list\x18\x04 \x01(\x0b\x32\x17.jcn.Model'
    b'.NeighborList\x12\x17\n\nunit_style\x18\x05 \x01(\tH\x00\x88\x01\x01\x12'
    b'\x12\n\nquantities\x18\x06 \x03(\t\x1a\x9b\x01\n\x0cNeighborList\x12)\n'
    b'\x04type\x18\x01 \x01(\x0e\x32\x1b.jcn.Model.NeighborListType\x12\x0e\n'
    b'\x06\x63utoff\x18\x02 \x01(\x02\x12\x11\n\tnbr_order\x18\x03 \x03(\x05'
    b'\x12\x16\n\thalf_list\x18\x04 \x01(\x08H\x00\x88\x01\x01\x12\x17\n\x0f'
    b'statistics_keys\x18\x05 \x03(\tB\x0c\n\n_half_list\"[\n\x10NeighborListType'
    b'\x12\x0f\n\x0bUNSPECIFIED\x10\x00\x12\x11\n\rDEVICE_SPARSE\x10\x01\x12'
    b'\x10\n\x0cSIMPLE_DENSE\x10\x02\x12\x11\n\rSIMPLE_SPARSE\x10\x03\x42\r\n'
    b'\x0b_unit_styleb\x06proto3'
)

# Build message classes via reflection (compatible with protobuf 3.20+)
_MODEL = DESCRIPTOR.message_types_by_name['Model']
_MODEL_NEIGHBORLIST = _MODEL.nested_types_by_name['NeighborList']

Model = _reflection.GeneratedProtocolMessageType('Model', (_message.Message,), {
    'DESCRIPTOR': _MODEL,
    'NeighborList': _reflection.GeneratedProtocolMessageType('NeighborList', (_message.Message,), {
        'DESCRIPTOR': _MODEL_NEIGHBORLIST,
        '__module__': 'bam.lammps.deploy.model_pb2',
    }),
    '__module__': 'bam.lammps.deploy.model_pb2',
})

_sym_db.RegisterMessage(Model)
_sym_db.RegisterMessage(Model.NeighborList)

# Expose the enum values at the Model class level for compatibility
# with chemtrain's usage: model_proto.Model.NeighborListType.SIMPLE_SPARSE
Model.NeighborListType = _MODEL.enum_types_by_name['NeighborListType']
Model.UNSPECIFIED = 0
Model.DEVICE_SPARSE = 1
Model.SIMPLE_DENSE = 2
Model.SIMPLE_SPARSE = 3
