import zipdir from 'zip-dir'

const sourceDir = process.argv[2] || './dist'
const outputPath = process.argv[3] || './dist.zip'

zipdir(sourceDir, { saveTo: outputPath }, function (err, buffer) {
  if (err) {
    console.error(`Error zipping "${sourceDir}" directory:`, err)
  } else {
    process.stdout.write(
      `Successfully zipped "${sourceDir}" directory to "${outputPath}".\n`
    )
  }
})
