export enum LatentPreviewMethod {
  NoPreviews = 'none',
  Auto = 'auto',
  Latent2RGB = 'latent2rgb',
  TAESD = 'taesd'
}

export enum LogLevel {
  DEBUG = 'DEBUG',
  INFO = 'INFO',
  WARNING = 'WARNING',
  ERROR = 'ERROR',
  CRITICAL = 'CRITICAL'
}

export enum HashFunction {
  MD5 = 'md5',
  SHA1 = 'sha1',
  SHA256 = 'sha256',
  SHA512 = 'sha512'
}

export enum CudaMalloc {
  // Let server decide whether to use CUDA malloc based on the current environment
  Auto = 'auto',
  // Disable CUDA malloc
  Disable = 'disable',
  // Enable CUDA malloc
  Enable = 'enable'
}

export enum FloatingPointPrecision {
  AUTO = 'auto',
  FP64 = 'fp64',
  FP32 = 'fp32',
  FP16 = 'fp16',
  BF16 = 'bf16',
  FP8E4M3FN = 'fp8_e4m3fn',
  FP8E5M2 = 'fp8_e5m2'
}

export enum CrossAttentionMethod {
  Auto = 'auto',
  Split = 'split',
  Quad = 'quad',
  Pytorch = 'pytorch'
}

export enum VramManagement {
  Auto = 'auto',
  GPUOnly = 'gpu-only',
  HighVram = 'highvram',
  NormalVram = 'normalvram',
  LowVram = 'lowvram',
  NoVram = 'novram',
  CPU = 'cpu'
}
