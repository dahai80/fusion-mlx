class FusionMlx < Formula
    desc "Unified local model management for Apple Silicon"
    homepage "https://github.com/dahai80/fusion-mlx"
    url "https://github.com/dahai80/fusion-mlx/archive/refs/tags/v0.6.13.tar.gz"
    sha256 "PLACEHOLDER_SHA256"
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
        sha256 "PLACEHOLDER_MLX_LM_SHA256"
    end

    resource "mlx-embeddings" do
        url "https://github.com/Blaizzy/mlx-embeddings.git",
            commit: "32981fa4e8064ed664b52071789dd18271fe4206"
        sha256 "PLACEHOLDER_MLX_EMB_SHA256"
    end

    resource "mlx-vlm" do
        url "https://github.com/Blaizzy/mlx-vlm.git",
            commit: "f96138eef1f5ce7fb5d97f8dd41a664a195b5659"
        sha256 "PLACEHOLDER_MLX_VLM_SHA256"
    end

    resource "dflash-mlx" do
        url "https://github.com/bstnxbt/dflash-mlx.git",
            commit: "1ba671372b289c025b435c1a13aabb4bfb80b183"
        sha256 "PLACEHOLDER_DFLASH_SHA256"
    end

    resource "mlx-audio" do
        url "https://github.com/Blaizzy/mlx-audio.git",
            commit: "51753266e0a4f766fd5e6fbc46652224efc23981"
        sha256 "PLACEHOLDER_MLX_AUDIO_SHA256"
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
