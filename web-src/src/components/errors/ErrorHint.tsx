export function ErrorHint({script}: { script: string }) {
    return (
        <>
            <span>This may be due to the following script:</span>
            <br/>
            <span style={{fontWeight: 'bold'}}>{script}</span>
        </>
    )
}
