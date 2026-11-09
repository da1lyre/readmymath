FROM ubuntu:22.04 AS builder

RUN apt-get update && apt-get install -y \
    build-essential cmake git curl ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --recursive https://github.com/CrispStrobe/CrispEmbed /src/CrispEmbed
WORKDIR /src/CrispEmbed

RUN cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCRISPEMBED_BUILD_SHARED=ON \
    && cmake --build build -j$(nproc) --target crispembed-shared crispembed-server

FROM python:3.10-slim

RUN apt-get update && apt-get install -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/CrispEmbed/build /opt/crispembed/build
RUN find /opt/crispembed/build -name 'lib*.so*' -exec cp -a {} /usr/local/lib/ \; && ldconfig

ENV PATH=/opt/crispembed/build:$PATH
ENV CRISPEMBED_CACHE_DIR=/cache

WORKDIR /app

RUN pip install --upgrade pip
COPY infra/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

COPY webapp/ ./webapp/
COPY src/ ./src/

CMD ["uvicorn", "webapp.api:app", "--host", "0.0.0.0", "--port", "8000"]