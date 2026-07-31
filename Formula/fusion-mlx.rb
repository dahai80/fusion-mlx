# SPDX-License-Identifier: Apache-2.0
# Homebrew formula for fusion-mlx
# Usage:
#   brew tap dahai80/fusion-mlx https://github.com/dahai80/fusion-mlx
#   brew install fusion-mlx
#
# Or one-liner:
#   brew install dahai80/fusion-mlx/fusion-mlx

class FusionMlx < Formula
    include Language::Python::Virtualenv

    desc "AI inference for Apple Silicon — OpenAI-compatible server, chat REPL, model management"
    homepage "https://github.com/dahai80/fusion-mlx"
    url "https://github.com/dahai80/fusion-mlx/archive/refs/tags/v0.6.1.tar.gz"
    sha256 "SKIP_SHA256_CHECK"
    license "Apache-2.0"
    head "https://github.com/dahai80/fusion-mlx.git", branch: "main"

    depends_on "python@3.13"

    on_macos do
        depends_on macos: :ventura
    end

    resource "fusion-mlx-pip" do
        url "https://pypi.org/pypi/fusion-mlx/json"
    end

    def install
        virtualenv_install_with_resources
    end

    test do
        assert_match version.to_s, shell_output("#{bin}/fusion-mlx --version")
    end
end
