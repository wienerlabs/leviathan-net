# Trains a short run with real clients against a local centralized server, and
# returns once every client has seen the run through to its end and returned
# cleanly. Sourced by the scripts that check what real clients do.
#
# The caller sets:
#   LABEL         the prefix of what this prints, e.g. "[fidelity]"
#   RUN_CONFIG    the run's config, which names its run id
#   WORK          where the logs, dumps and states go
#   BIN           where the server and client binaries are
#   PORT          the port the server listens on
#   TIMEOUT_SECS  how long the run may take before it counts as stuck
#   NODES         how many clients train it
#
# Client N keeps every payload in "$WORK/dumpsN" (--write-gradients-dir) and
# the state each round trains from in "$WORK/statesN" (--round-state-dir), and
# logs to "$WORK/clientN.log".

# macOS strips DYLD_* when it execs a protected shell, so a caller's exported
# library path never reaches these binaries. Rebuild it here instead.
TORCH_VENV="${TORCH_VENV:-$HOME/.leviathan-torch}"
if [[ -d "$TORCH_VENV" ]]; then
    TORCH_LIB=$(find "$TORCH_VENV" -maxdepth 6 -type d -path "*site-packages/torch/lib" | head -1)
    PYTHON_LIB="${PYTHON_LIB:-/opt/homebrew/opt/python@3.13/Frameworks/Python.framework/Versions/3.13/lib}"
    if [[ -n "$TORCH_LIB" ]]; then
        export DYLD_LIBRARY_PATH="$TORCH_LIB:$PYTHON_LIB${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
    fi
fi

PIDS=()
cleanup() {
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT

# A client returns once its run is finished and its last checkpoint is done
# (wienerlabs/leviathan-net#82). It has to have seen the run through to the end
# and returned cleanly. The console log colours its fields, so the escapes go
# first.
client_finished() {
    tr -d '\033' < "$WORK/client$1.log" | grep -aq 'new_state[^ ]*Finished'
}

run_real_clients() {
    echo "$LABEL work dir $WORK"
    "$BIN/psyche-centralized-server" run --state "$RUN_CONFIG" --server-port "$PORT" --tui false \
        > "$WORK/server.log" 2>&1 &
    local server=$!
    PIDS+=($server)
    # The clients connect once the server listens, not after a guess at how
    # long that takes; a port someone else holds stops the server here.
    local ready=$((SECONDS + 60))
    until grep -aq "Server listening on" "$WORK/server.log"; do
        if ! kill -0 "$server" 2>/dev/null || ((SECONDS > ready)); then
            echo "$LABEL the server did not start on port $PORT; logs in $WORK" >&2
            exit 1
        fi
        sleep 1
    done

    local run_id
    run_id="$(sed -n 's/^run_id = "\(.*\)"$/\1/p' "$RUN_CONFIG")"
    local clients=()
    local returned=()
    local node
    for ((node = 0; node < NODES; node++)); do
        # A fresh identity per client: the run admits each by its own key.
        RAW_IDENTITY_SECRET_KEY=$(od -An -N32 -tx1 /dev/urandom | tr -d ' \n') \
            "$BIN/psyche-centralized-client" train \
            --run-id "$run_id" \
            --server-addr "localhost:$PORT" --device cpu --logs console \
            --iroh-relay disabled --iroh-discovery local \
            --write-gradients-dir "$WORK/dumps$node" \
            --round-state-dir "$WORK/states$node" \
            > "$WORK/client$node.log" 2>&1 &
        PIDS+=($!)
        clients+=($!)
    done

    echo "$LABEL $NODES clients training $RUN_CONFIG"
    local deadline=$((SECONDS + TIMEOUT_SECS))
    while true; do
        local running=0
        for ((node = 0; node < NODES; node++)); do
            if kill -0 "${clients[$node]}" 2>/dev/null; then
                running=$((running + 1))
            elif [[ -z "${returned[$node]:-}" ]]; then
                local status=0
                wait "${clients[$node]}" || status=$?
                if ((status != 0)) || ! client_finished "$node"; then
                    echo "$LABEL client $node stopped with status $status before it saw the run finish; logs in $WORK" >&2
                    exit 1
                fi
                returned[$node]=1
            fi
        done
        if ((running == 0)); then
            break
        fi
        if ((SECONDS > deadline)); then
            echo "$LABEL the run did not finish in ${TIMEOUT_SECS}s; logs in $WORK" >&2
            exit 1
        fi
        if ! kill -0 "$server" 2>/dev/null; then
            echo "$LABEL the server stopped before the run finished; logs in $WORK" >&2
            exit 1
        fi
        # A client dropped from the run leaves the others waiting for members
        # for good, and its rounds can no longer be accounted for.
        if grep -aq "Dropped .* clients" "$WORK/server.log"; then
            echo "$LABEL the server dropped a client; logs in $WORK" >&2
            exit 1
        fi
        sleep 5
    done
}
