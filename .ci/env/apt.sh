#!/bin/bash
#===============================================================================
# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================
# apt.sh文件通常用于定义项目的构建和测试环境所需的系统依赖项，例如编译工具、库文件、运行时环境等
# 文件内容：配置环境和安装依赖项的 Bash 脚本
# 目的：在不同的情况下配置开发环境和安装特定的工具和库

component=$1

function update {
    # 更新系统的软件包列表
    sudo apt-get update
}

function add_repo {
    # 使系统能识别并获取来自 Intel oneAPI 仓库的软件包
    # 具体实现    下载并添加 Intel oneAPI 的 GPG 公钥
    #             将 oneAPI 仓库的源信息添加到系统中
    #             更新软件包列表确保仓库信息已被成功加载
    wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB
    sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB
    rm GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB
    echo "deb https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
    sudo add-apt-repository -y "deb https://apt.repos.intel.com/oneapi all main"
    sudo apt-get update
}

function install_dpcpp {
    sudo apt-get install -y intel-dpcpp-cpp-compiler-2023.2.1
    sudo bash -c 'echo libintelocl.so > /etc/OpenCL/vendors/intel-cpu.icd'
    sudo mv -f /opt/intel/oneapi/compiler/latest/linux/lib/oclfpga /opt/intel/oneapi/compiler/latest/linux/lib/oclfpga_
}

function install_mkl {
    sudo apt-get install intel-oneapi-mkl-devel
}

function install_clang-format {
    sudo apt-get install -y clang-format-14
    sudo update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-14 100
    sudo update-alternatives --set clang-format /usr/bin/clang-format-14
}

function install_dev-base {
    sudo apt-get install -y gcc-multilib g++-multilib dos2unix tree
}

function install_dev-base-conda {
    conda env create -f .ci/env/environment.yml
}

if [ "${component}" == "dpcpp" ]; then
    add_repo
    install_dpcpp
elif [ "${component}" == "mkl" ]; then
    add_repo
    install_mkl
elif [ "${component}" == "clang-format" ]; then
    update
    install_clang-format
elif [ "${component}" == "dev-base" ]; then
    update
    install_dev-base
    install_dev-base-conda
else
    echo "Usage:"
    echo "   $0 [dpcpp|mkl|clang-format|dev-base]"
    exit 1
fi
