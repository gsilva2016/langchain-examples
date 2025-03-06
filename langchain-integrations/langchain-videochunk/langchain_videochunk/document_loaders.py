"""VideoChunkLoader document loader."""

import os
import shutil
import subprocess
from typing import AsyncIterator, Dict, Iterator, List

from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document


class VideoChunkLoader(BaseLoader):
    """
    VideoChunkLoader document loader integration

    Setup:
        Install ``langchain-videochunk`` 

        .. code-block:: bash

            pip install -U langchain-videochunk

    Instantiate:
        .. code-block:: python

            from langchain_community.document_loaders import VideoChunkLoader

            loader = VideoChunkLoader(
                video_path="video_file.mp4",
                chunking_mechanism="sliding_window" # specific
                chunk_duration=10,
                chunk_overlap=2
            )

    Lazy load:
        .. code-block:: python

            docs = loader.lazy_load()

            for doc in docs:
                docs.append(doc)
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "Chunk metadata: {'chunk_id': 0, 'chunk_path': 'video_chunks/chunk_0.mp4', 'start_time': 10, 'end_time': 20, 'source': 'sample_video.mp4'}\n",
                        "Chunk content: Video chunk from 10s to 20s\n",
                        "Chunk metadata: {'chunk_id': 1, 'chunk_path': 'video_chunks/chunk_1.mp4', 'start_time': 20, 'end_time': 28, 'source': 'sample_video.mp4'}\n",
                        "Chunk content: Video chunk from 20s to 28s\n"
                    ]
                }
            ]

    Async load:
        .. code-block:: python

            docs = await loader.alazy_load()
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "Chunk metadata: {'chunk_id': 0, 'chunk_path': 'video_chunks/chunk_0.mp4', 'start_time': 10, 'end_time': 20, 'source': 'sample_video.mp4'}\n",
                        "Chunk content: Video chunk from 10s to 20s\n",
                        "Chunk metadata: {'chunk_id': 1, 'chunk_path': 'video_chunks/chunk_1.mp4', 'start_time': 20, 'end_time': 28, 'source': 'sample_video.mp4'}\n",
                        "Chunk content: Video chunk from 20s to 28s\n"
                    ]
                }
            ]
    """  # noqa: E501

    def __init__(
        self,
        video_path: str,
        # "specific_chunks" or "sliding_window"
        chunking_mechanism: str = "sliding_window",
        chunk_duration: int = 10,
        chunk_overlap: int = 2,
        specific_intervals: List[Dict] = [],
        output_dir: str = "video_chunks",
    ) -> None:
        """Initialize the loader with video chunking parameters.
        Args:
            video_path: Path to the video file.
            chunking_mechanism: "sliding_window" or "specific_chunks".
            chunk_duration: Duration of each chunk in seconds (for sliding window).
            chunk_overlap: Between consecutive chunks in seconds (for sliding window).
            specific_intervals: List of specific intervals for chunking.
            output_dir: Directory to save chunked videos.
        """
        self.video_path = video_path
        self.chunking_mechanism = chunking_mechanism
        self.chunk_duration = chunk_duration
        self.chunk_overlap = chunk_overlap
        self.specific_intervals = specific_intervals
        self.output_dir = output_dir

        if self.output_dir:
            # Remove the existing directory if it alreade exists
            # and create a fresh one before processing new chunks
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)
            os.makedirs(self.output_dir)

    def _compute_sliding_window_intervals(self) -> List[Dict]:
        """Compute intervals for sliding window chunking."""
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                self.video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        duration = float(result.stdout.strip())
        intervals = []
        start_time = 0

        while start_time < duration:
            end_time = min(start_time + self.chunk_duration, duration)
            intervals.append({"start": start_time, "end": end_time})
            start_time += self.chunk_duration - self.chunk_overlap

        return intervals

    def _compute_specific_intervals(self) -> Dict[int, Dict]:
        """Compute intervals for specific chunking."""
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                self.video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        video_duration = float(result.stdout.strip())

        intervals = {}
        for idx, interval in enumerate(self.specific_intervals):
            start_time = interval["start"]
            duration = interval["duration"]

            if start_time < 0 or start_time >= video_duration:
                raise ValueError(
                    f"start_time {start_time} is out of bounds for the video duration."
                )
            if duration <= 0:
                raise ValueError("duration must be greater than 0.")
            if start_time + duration > video_duration:
                raise ValueError("The specified interval exceeds the video duration.")

            intervals[idx] = {"start": start_time, "end": start_time + duration}

        return intervals

    def _save_video_chunk(
        self, start_time: float, duration: float, chunk_id: int
    ) -> str:
        """Save a video chunk using ffmpeg."""

        output_path = os.path.join(self.output_dir, f"chunk_{chunk_id}.mp4")
        command = [
            "ffmpeg",
            "-ss",
            str(start_time),
            "-t",
            str(duration),
            "-i",
            self.video_path,
            "-c",
            "copy",  # Copy codec for fast processing
            output_path,
        ]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_path

    def lazy_load(self) -> Iterator[Document]:
        """Lazily load video chunks as langchain Documents."""
        if self.chunking_mechanism == "specific_chunks":
            si_intervals = self._compute_specific_intervals()
        else:
            wi_intervals = self._compute_sliding_window_intervals()

        for chunk_id, interval in enumerate(
            si_intervals.values()
            if self.chunking_mechanism == "specific_chunks"
            else wi_intervals
        ):
            start_time = interval["start"]
            duration = interval["end"] - interval["start"]
            chunk_path = self._save_video_chunk(start_time, duration, chunk_id)
            yield Document(
                page_content=f"Video chunk from {start_time}s to {interval['end']}s",
                metadata={
                    "chunk_id": chunk_id,
                    "chunk_path": chunk_path,  # Path to the saved video chunk
                    "start_time": start_time,
                    "end_time": interval["end"],
                    "source": self.video_path,  # Original video file path
                },
            )

    async def alazy_load(self) -> AsyncIterator[Document]:
        """Async lazy loader for video chunks."""
        for doc in self.lazy_load():
            yield doc
