#!/bin/bash


nohup python train.py --loss_type crossentropy --weighted_sampler --early_stopping &
