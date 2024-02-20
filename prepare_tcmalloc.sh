#!/usr/bin/env bash
#################################################
# Quoted from Stable Diffusion WEBUI            #
#################################################
prepare_tcmalloc() {
    if [[ "${OSTYPE}" == "linux"* ]] && [[ -z "${LD_PRELOAD}" ]]; then
        # check glibc version
        LIBC_VER=$(echo $(ldd --version | awk 'NR==1 {print $NF}') | grep -oP '\d+\.\d+')
        echo "glibc version is $LIBC_VER"
        libc_vernum=$(expr $LIBC_VER)
        # Since 2.34 libpthread is integrated into libc.so
        libc_v234=2.34
        # Define Tcmalloc Libs arrays
        TCMALLOC_LIBS=("libtcmalloc(_minimal|)\.so\.\d" "libtcmalloc\.so\.\d")
        # Traversal array
        for lib in "${TCMALLOC_LIBS[@]}"
        do
            # Determine which type of tcmalloc library the library supports
            TCMALLOC="$(PATH=/usr/sbin:$PATH ldconfig -p | grep -P $lib | head -n 1)"
            TC_INFO=(${TCMALLOC//=>/})
            if [[ ! -z "${TC_INFO}" ]]; then
                echo "Check TCMalloc: ${TC_INFO}"
                # Determine if the library is linked to libpthread and resolve undefined symbol: pthread_key_create
                if [ $(echo "$libc_vernum < $libc_v234" | bc) -eq 1 ]; then
                    # glibc < 2.34 pthread_key_create into libpthread.so. check linking libpthread.so...
                    if ldd ${TC_INFO[2]} | grep -q 'libpthread'; then
                        echo "$TC_INFO is linked with libpthread,execute LD_PRELOAD=${TC_INFO[2]}"
                        # set fullpath LD_PRELOAD (To be on the safe side)
                        export LD_PRELOAD="${TC_INFO[2]}"
                        break
                    else
                        echo "$TC_INFO is not linked with libpthread will trigger undefined symbol: pthread_Key_create error"
                    fi
                else
                    # Version 2.34 of libc.so (glibc) includes the pthread library IN GLIBC. (USE ubuntu 22.04 and modern linux system and WSL)
                    # libc.so(glibc) is linked with a library that works in ALMOST ALL Linux userlands. SO NO CHECK!
                    echo "$TC_INFO is linked with libc.so,execute LD_PRELOAD=${TC_INFO[2]}"
                    # set fullpath LD_PRELOAD (To be on the safe side)
                    export LD_PRELOAD="${TC_INFO[2]}"
                    break
                fi
            fi
        done
        if [[ -z "${LD_PRELOAD}" ]]; then
            printf "\e[1m\e[31mCannot locate TCMalloc. Do you have tcmalloc or google-perftool installed on your system? (improves CPU memory usage)\e[0m\n"
        fi
    fi
}
prepare_tcmalloc
