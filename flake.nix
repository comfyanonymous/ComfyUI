{
  description = "ComfyUI with requirements.txt";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    pyproject-nix.url = "github:pyproject-nix/pyproject.nix";
  };

  outputs = {
    self,
    nixpkgs,
    pyproject-nix,
    ...
  } @ inputs: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config.allowUnfree = true;
      config.cudaSupport = true;
    };
    python = let
      packageOverrides = self: super: {
        "comfyui-frontend-package" =
			let version="1.25.5";
		in
          super.buildPythonPackage
          {
            pname = "comfyui-frontend-package";
			inherit version;
            pyproject = true;
            src = fetchTarball {
              url = "https://files.pythonhosted.org/packages/f6/1f/07a491e28dd754297ac2bc74579599efde1c3be233ed209f2d355d5268e2/comfyui_frontend_package-1.25.5.tar.gz";
              sha256 = "1dvs8irqz9wpq314khxigp63ahqxrsb9yhhbv3gj13bxznybc1w7";
            };
            propagatedBuildInputs = with super; [
              setuptools
              torch
              torchvision
              safetensors
              numpy
              einops
              typing-extensions
            ];
            COMFYUI_FRONTEND_VERSION = version;
          };
        "spandrel" = super.buildPythonPackage {
          pname = "spandrel";
          pyproject = true;
          version = "0.4.1";
          src = fetchTarball {
            url = "https://files.pythonhosted.org/packages/45/e0/048cd03119a9f2b685a79601a52311d5910ff6fd710c01f4ed6769a2892f/spandrel-0.4.1.tar.gz";
            sha256 = "1byaq7mzjs27qhs9d7aw6xflqlj3qzm47mgawdxrlcw9qj4gk9w4";
          };
          buildInputs = with super; [
            torch
            torchvision
            safetensors
            einops
          ];
        };
        "gguf-node" = super.buildPythonPackage {
          pname = "gguf-node";
          pyproject = true;
          version = "0.2.5";
          src = fetchTarball {
            url = "https://files.pythonhosted.org/packages/44/b3/9e30eb328326ab03cce830ff48700d23315c663b9dbac26ec63296f198e4/gguf_node-0.2.5.tar.gz";
            sha256 = "1bhl2nik3i1ni7hi2z57skndip9599nij2i2y4q6z6qmizm0wvva";
          };
          buildInputs = with super; [
            flit-core
            tqdm
          ];
        };
        "clip-interrogator" = super.buildPythonPackage {
          pname = "clip-interrogator";
          version = "0.6.0";
          src = fetchTarball {
            url = "https://files.pythonhosted.org/packages/23/d1/2f0f61c5cbaea3d1480f2eb2709f89d64d62976e9634e7eeaac2e2c03ba2/clip-interrogator-0.6.0.tar.gz";
            sha256 = "04jgxjy58n9nlr5qvjiqrn0n3rm9m113c9a8y882qjckha53hd2z";
          };
        };
        "comfy-cli" = super.buildPythonPackage {
          pname = "comfy-cli";
          version = "1.3.8";
          pyproject = true;
          src = fetchTarball {
            url = "https://files.pythonhosted.org/packages/12/00/f07a30796085324d0d5aab49a2cff0e504e65bf9e69e5a0a6046933e45ff/comfy_cli-1.3.8.tar.gz";
            sha256 = "0kr3rinrdlnkwwvk4dh43wlm2dgjpxrnyqwlikg5fjsg2nscmbym";
          };
          propagatedBuildInputs = with super; [
            setuptools
            charset-normalizer
            gitpython
            httpx
            mixpanel
            pathspec
            psutil
            pyyaml
            questionary
            requests
            rich
            tomlkit
            typer
            typing-extensions
            uv
            websocket-client
            ruff
            semver
            cookiecutter
          ];
        };
      };
    in
      pkgs.python3.override {
        inherit packageOverrides;
        self = python;
      };
    project = pyproject-nix.lib.project.loadRequirementsTxt {
      projectRoot = ./.;
    };
    pythonEnv = python.withPackages (project.renderers.withPackages {
      inherit python;
    });
  in {
    formatter.${system} = nixpkgs.legacyPackages.${system}.treefmt;
    devShells.${system}.default = pkgs.mkShell {
      packages = [
        pythonEnv
      ];
    };
  };
}
