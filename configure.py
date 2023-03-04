import cmd, requests, os

class ComfyConfigure(cmd.Cmd):
    intro = "Welcome to ComfyUI configure shell. Type help or ? to list commands.\n"
    prompt = "(configure) "
    file = None

    def do_install_esrgan_deps(self, arg):
        'Install base ESRGAN/GFPGAN model dependencies'
        self.install_model('realesrgan', 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth')
        self.install_model('realesrgan', 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth')
        self.install_model('gfpgan', 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth')
        print('done!')

    def do_exit(self, arg):
        'Exit the shell'
        return True
    
    def install_model(self, category, url):
        models_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", category)
        if not os.path.isdir(models_dir):
            os.mkdir(models_dir)

        print('downloading {0}...'.format(url))
        resp = requests.get(url)
        if resp:
            with open(os.path.join(models_dir, os.path.basename(url)), "wb") as file:
                file.write(resp.content)
        else:
            print('failed to download {0}: {1}', url, resp.text)

if __name__ == '__main__':
    ComfyConfigure().cmdloop()
