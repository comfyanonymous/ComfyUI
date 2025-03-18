{
  description = "ComfyUI with requirements.txt";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
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
      packageOverrides = self: super:
      {
        "comfyui-frontend-package" = super.buildPythonPackage 
        {
          pname = "comfyui-frontend-package";
          version = "1.12.14";
          src = fetchTarball {
            url = "https://files.pythonhosted.org/packages/cc/33/0ca463657227a7538b8b1d924840e4e25307f8c8d20c683439d7ea97ba6d/comfyui_frontend_package-1.12.14.tar.gz";
            sha256 = "1c5j01xqkzxqzj5nf646jrs7g095b0pzliy9b3b60wrspkaa2296";
          };
          propagatedBuildInputs = [
            super.setuptools
          ];
          COMFYUI_FRONTEND_VERSION = "1.12.14";
        };
        "spandrel" = super.buildPythonPackage {
          pname = "spandrel";
          pyproject = true;
          version = "0.4.1";
          src = fetchTarball {
            url = "https://files.pythonhosted.org/packages/45/e0/048cd03119a9f2b685a79601a52311d5910ff6fd710c01f4ed6769a2892f/spandrel-0.4.1.tar.gz";
            sha256 = "1byaq7mzjs27qhs9d7aw6xflqlj3qzm47mgawdxrlcw9qj4gk9w4";
          };
          buildInputs = [
            super.torch
            super.torchvision
            super.safetensors
            super.einops
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
    pythonEnv = assert project.validators.validateVersionConstraints {inherit python;} == {}; (
      python.withPackages (project.renderers.withPackages {
        inherit python;
      })
    );
  in {
    formatter.${system} = nixpkgs.legacyPackages.${system}.alejandra;
    devShells.${system}.default = pkgs.mkShell {
      packages = [
        pythonEnv
        #(python.withPackages (ps: with ps; [ comfyui-frontend-package ]))
      ];
    };
  };
}
