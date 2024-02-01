import {ReactNode} from "react";

interface WorkflowLoadErrorProps {
    err: Error,
    errorHint: ReactNode[]
}

export function WorkflowLoadError({err, errorHint}: WorkflowLoadErrorProps) {
    return (
        <div>
            <p>Loading aborted due to error reloading workflow data</p>
            <pre style={{
                padding: '5px',
                backgroundColor: 'rgba(255,0,0,0.2)'
            }}>
                {err.toString()}
            </pre>

            <pre style={{
                padding: '5px',
                color: '#ccc',
                fontSize: '10px',
                maxHeight: '50vh',
                overflow: 'auto',
                backgroundColor: 'rgba(0,0,0,0.2)',
            }}>
                {err.stack || 'No stacktrace available'}
            </pre>

            {
                errorHint.map(
                    (hint, index) => <div key={index}>{hint}</div>
                )
            }
        </div>
    )
}
