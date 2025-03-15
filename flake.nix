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
    python = pkgs.python3;
    project = pyproject-nix.lib.project.loadRequirementsTxt {
      projectRoot = ./.;
    };
    pythonEnv = assert project.validators.validateVersionConstraints {inherit python;} == {}; (
      pkgs.python3.withPackages (project.renderers.withPackages {
        inherit python;
      })
    );
  in {
    formatter.${system} = nixpkgs.legacyPackages.${system}.alejandra;
    devShells.${system}.default = pkgs.mkShell {
      packages = [
        pythonEnv
      ];
    };
  };
}
