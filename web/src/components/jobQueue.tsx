// This class tracks state of the job-queue, and provides methods to add or remove jobs from it
// What is a job queue?
// It can be either (1) a list of jobs that are stored on the client, which the client
// will submit to the server sequentially.
// (2) a list of jobs which are already submitted to the server, and the client
// is subscribing for their results.
// The latter is more powerful; it allows (1) multiple clients to submit to the
// same queue (i.e., if you have multiple browsers open or are sharing a session with
// others), and (2) it allows the server to split jobs across multiple workers,
// because the GPU already knows what's going on.

import React, { useEffect, useState, ReactNode } from 'react';
import { ComfyCanvas } from '../litegraph/comfyCanvas';
import { useJobQueue } from '../context/jobQueueContext';
import { ComfyGraph } from '../litegraph/comfyGraph';

// This is a component that lists all jobs in the queue
export function JobQueueList(): ReactNode {
    const { queue, lastNodeErrors } = useJobQueue();

    return (
        <div className="job-queue">
            <div className="job-queue-items">
                {queue.map(item => (
                    <div className="job-queue-item" key={item.jobId}>
                        <span className="job-queue-status">{item.status}</span>
                    </div>
                ))}
            </div>
            
            {lastNodeErrors && (
                <div className="job-queue-errors">
                    <span className="job-queue-errors-title">Errors</span>
                    <div className="job-queue-errors-items">
                        {Object.keys(lastNodeErrors).map(key => (
                            <div className="job-queue-errors-item" key={key}>
                                <span className="job-queue-errors-item-node">{key}</span>
                                <span className="job-queue-errors-item-error">
                                    {lastNodeErrors[key].errors.join(', ')}
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
