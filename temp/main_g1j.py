# MIT License

from evaluation_runner import G1J_CAPACITIES, G1J_MANIFESTS, main


if __name__ == "__main__":
    raise SystemExit(main(default_manifests=G1J_MANIFESTS, expected_capacities=G1J_CAPACITIES))
