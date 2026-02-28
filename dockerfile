FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
SHELL ["/bin/bash", "-lc"]

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl tar gzip xz-utils \
 && rm -rf /var/lib/apt/lists/*

ENV PYHOME=/opt/python
COPY vad.tar.gz ${PYHOME}.tar.gz
RUN mkdir -p ${PYHOME} \
 && tar -xzf ${PYHOME}.tar.gz -C ${PYHOME} \
 && rm ${PYHOME}.tar.gz 


ENV PATH=${PYHOME}/bin:$PATH
ENV LD_LIBRARY_PATH=${PYHOME}/lib:${PYHOME}/lib64:$LD_LIBRARY_PATH
WORKDIR /workspace
CMD ["/bin/bash"]