#!/usr/bin/env bash

set -ex

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PY_ARGS=${@:4}

PROJECT_NAME=${PROJECT_NAME:-denseclip_test}
ENTRY_FILE=${ENTRY_FILE:-tools/test.py}
WORKBENCH=${WORKBENCH:-search_algo_quality_dev}  # search_algo_quality_dev, imac_dev
ROLEARN=${ROLEARN:-searchalgo}  # searchalgo, imac

tar -zchf /tmp/${PROJECT_NAME}.tar.gz --exclude work_dirs --exclude data --exclude local_data --exclude pretrained --exclude .git .
cmd_oss="
use ${WORKBENCH};
pai -name pytorch180
    -Dscript=\"file:///tmp/${PROJECT_NAME}.tar.gz\"
    -DentryFile=\"${ENTRY_FILE}\"
    -DworkerCount=${GPUS}
    -DuserDefinedParameters=\"${CONFIG} ${CHECKPOINT} --launcher pytorch ${PY_ARGS} --gpu-collect\"
    -Dbuckets=\"oss://mvap-data/zhax/wangluting/?role_arn=acs:ram::1367265699002728:role/${ROLEARN}4pai&host=cn-zhangjiakou.oss.aliyuncs.com\";
"
    # -Dcluster=\"{\\\"worker\\\":{\\\"gpu\\\":${GPUS}00}}\"
odpscmd -e "${cmd_oss}"
