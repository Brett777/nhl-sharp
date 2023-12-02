#!/usr/bin/env bash
#
#  Copyright 2023 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
echo "Starting App"
export OPENAI_API_KEY=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
streamlit run hockey.py
