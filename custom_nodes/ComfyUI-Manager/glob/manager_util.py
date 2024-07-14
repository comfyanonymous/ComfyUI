try:
    from distutils.version import StrictVersion
except:
    print(f"[ComfyUI-Manager]  'distutils' package not found. Activating fallback mode for compatibility.")
    class StrictVersion:
        def __init__(self, version_string):
            self.version_string = version_string
            self.major = 0
            self.minor = 0
            self.patch = 0
            self.pre_release = None
            self.parse_version_string()

        def parse_version_string(self):
            parts = self.version_string.split('.')
            if not parts:
                raise ValueError("Version string must not be empty")

            self.major = int(parts[0])
            self.minor = int(parts[1]) if len(parts) > 1 else 0
            self.patch = int(parts[2]) if len(parts) > 2 else 0

            # Handling pre-release versions if present
            if len(parts) > 3:
                self.pre_release = parts[3]

        def __str__(self):
            version = f"{self.major}.{self.minor}.{self.patch}"
            if self.pre_release:
                version += f"-{self.pre_release}"
            return version

        def __eq__(self, other):
            return (self.major, self.minor, self.patch, self.pre_release) == \
                (other.major, other.minor, other.patch, other.pre_release)

        def __lt__(self, other):
            if (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch):
                return self.pre_release_compare(self.pre_release, other.pre_release) < 0
            return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

        @staticmethod
        def pre_release_compare(pre1, pre2):
            if pre1 == pre2:
                return 0
            if pre1 is None:
                return 1
            if pre2 is None:
                return -1
            return -1 if pre1 < pre2 else 1

        def __le__(self, other):
            return self == other or self < other

        def __gt__(self, other):
            return not self <= other

        def __ge__(self, other):
            return not self < other

        def __ne__(self, other):
            return not self == other

