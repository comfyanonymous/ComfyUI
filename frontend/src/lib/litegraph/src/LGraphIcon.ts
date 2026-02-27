export interface LGraphIconOptions {
  unicode?: string
  fontFamily?: string
  image?: HTMLImageElement
  color?: string
  bgColor?: string
  fontSize?: number
  size?: number
  circlePadding?: number
  xOffset?: number
  yOffset?: number
}

export class LGraphIcon {
  unicode?: string
  fontFamily: string
  image?: HTMLImageElement
  color: string
  bgColor?: string
  fontSize: number
  size: number
  circlePadding: number
  xOffset: number
  yOffset: number

  constructor({
    unicode,
    fontFamily = 'PrimeIcons',
    image,
    color = '#e6c200',
    bgColor,
    fontSize = 16,
    size,
    circlePadding = 2,
    xOffset = 0,
    yOffset = 0
  }: LGraphIconOptions) {
    this.unicode = unicode
    this.fontFamily = fontFamily
    this.image = image
    this.color = color
    this.bgColor = bgColor
    this.fontSize = fontSize
    this.size = size ?? fontSize
    this.circlePadding = circlePadding
    this.xOffset = xOffset
    this.yOffset = yOffset
  }

  draw(ctx: CanvasRenderingContext2D, x: number, y: number) {
    x += this.xOffset
    y += this.yOffset

    if (this.image) {
      const iconSize = this.size
      const iconRadius = iconSize / 2 + this.circlePadding

      if (this.bgColor) {
        const { fillStyle } = ctx
        ctx.beginPath()
        ctx.arc(x + iconRadius, y, iconRadius, 0, 2 * Math.PI)
        ctx.fillStyle = this.bgColor
        ctx.fill()
        ctx.fillStyle = fillStyle
      }

      const imageX = x + this.circlePadding
      const imageY = y - iconSize / 2
      ctx.drawImage(this.image, imageX, imageY, iconSize, iconSize)
    } else if (this.unicode) {
      const { font, textBaseline, textAlign, fillStyle } = ctx

      ctx.font = `${this.fontSize}px '${this.fontFamily}'`
      ctx.textBaseline = 'middle'
      ctx.textAlign = 'center'
      const iconRadius = this.fontSize / 2 + this.circlePadding

      if (this.bgColor) {
        ctx.beginPath()
        ctx.arc(x + iconRadius, y, iconRadius, 0, 2 * Math.PI)
        ctx.fillStyle = this.bgColor
        ctx.fill()
      }

      ctx.fillStyle = this.color
      ctx.fillText(this.unicode, x + iconRadius, y)

      ctx.font = font
      ctx.textBaseline = textBaseline
      ctx.textAlign = textAlign
      ctx.fillStyle = fillStyle
    }
  }
}
