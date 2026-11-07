class FusionMlx < Formula
    desc "Unified local model management for Apple Silicon"
    homepage "https://github.com/dahai80/fusion-mlx"
    url "https://github.com/dahai80/fusion-mlx/archive/refs/tags/v0.10.52.tar.gz"
    sha256 "85f5f8049affaa5cbee022888fa5bd626f07caae972f6ebac28e4ec6de938c99"
    license "Apache-2.0"
    head "https://github.com/dahai80/fusion-mlx.git", branch: "main"

    depends_on "python@3.12"

    # ARM64-native wheels — must be pre-built for macOS arm64
    resource "mlx" do
        url "https://files.pythonhosted.org/packages/py3/m/mlx/mlx-0.32.0-cp312-cp312-macosx_14_0_arm64.whl"
        sha256 "ea5a594355c89c0095eaba413fd39d4caa8642fa13432dfb0c9354d141046467"
    end

    resource "safetensors" do
        url "https://files.pythonhosted.org/packages/py3/s/safetensors/safetensors-0.5.3-cp38-abi3-macosx_11_0_arm64.whl"
        sha256 "21d01c14ff6c415c485616b8b0bf961c46b3b343ca59110d38d744e577f9cce7"
    end

    # Git-pinned MLX ecosystem — pinned to verified commits
    resource "mlx-lm" do
        url "https://github.com/ml-explore/mlx-lm.git",
            commit: "ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd"
        sha256 "67e1a52f9b86551a24eab1aa2681c26a391819925bb10d14792b96a88303ebc7"
    end

    resource "mlx-embeddings" do
        url "https://github.com/Blaizzy/mlx-embeddings.git",
            commit: "32981fa4e8064ed664b52071789dd18271fe4206"
        sha256 "449c2c2fba27aff135b156985e0a9de18155143cc7c1ce1b84f284481c34a00b"
    end

    resource "mlx-vlm" do
        url "https://github.com/Blaizzy/mlx-vlm.git",
            commit: "f96138eef1f5ce7fb5d97f8dd41a664a195b5659"
        sha256 "d7c6c949f2291fac78203da0b3c354be4bc098d1dcd3dd80c8c92d03794c8ae3"
    end

    resource "dflash-mlx" do
        url "https://github.com/bstnxbt/dflash-mlx.git",
            commit: "1ba671372b289c025b435c1a13aabb4bfb80b183"
        sha256 "2098fdd3bb82bac1105408968e17d70fa34dde64f1a75682d81e77d531ca0fec"
    end

    resource "mlx-audio" do
        url "https://github.com/Blaizzy/mlx-audio.git",
            commit: "51753266e0a4f766fd5e6fbc46652224efc23981"
        sha256 "7f9297a18f4cfa8a30efde3ba0056b87dfbb5d64747591eef5ff44333f9a19ef"
    end

    def install
        virtualenv_install_with_resources
    end

    def caveats
        <<~EOS
            To start fusion-mlx as a launchd service:
              brew services start fusion-mlx

            Or run manually:
              fusion-mlx serve <model> --port 8000

            Model files are stored in ~/.fusion-mlx/models/
        EOS
    end

    service do
        run [opt_bin/"fusion-mlx", "serve", "--port", "8000"]
        keep_alive true
        run_at_load true
        environment_variables HF_ENDPOINT: "https://hf-mirror.com"
        log_path var/"log/fusion-mlx.log"
        error_log_path var/"log/fusion-mlx.err.log"
    end

    test do
        assert_match version.to_s, shell_output("#{bin}/fusion-mlx --version")
    end
end
