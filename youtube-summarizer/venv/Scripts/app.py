import streamlit as st
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import re
from nltk.tokenize import sent_tokenize
import nltk

nltk.download('punkt', quiet=True)


class LectureNoteGenerator:
    def __init__(self):
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    def extract_video_id(self, url):
        video_id = re.findall(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
        return video_id[0] if video_id else None

    def get_video_info(self, url):
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
                return {
                    'title': info.get('title', 'Untitled Video'),
                    'duration': info.get('duration', 0),
                    'description': info.get('description', '')
                }
            except Exception as e:
                st.error(f"Error getting video info: {str(e)}")
                return None

    def get_transcript(self, video_id):
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            return ' '.join([entry['text'] for entry in transcript])
        except Exception as e:
            st.error(f"Error getting transcript: {str(e)}")
            return None

    def generate_lecture_notes(self, text):
        chunks = self._chunk_text(text)
        summaries = self._summarize_chunks(chunks)
        return self._structure_notes(summaries)

    def _chunk_text(self, text, chunk_size=1000):
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            if current_size + len(sentence) > chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = len(sentence)
            else:
                current_chunk.append(sentence)
                current_size += len(sentence)

        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks

    def _summarize_chunks(self, chunks):
        summaries = []
        for chunk in chunks:
            summary = self.summarizer(chunk, max_length=150, min_length=50, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        return summaries

    def _structure_notes(self, summaries):
        intro = summaries[0]
        body = summaries[1:-1]
        conclusion = summaries[-1]

        notes = {
            "Introduction": intro,
            "Main Content": body,
            "Conclusion": conclusion
        }
        return notes


def main():
    st.title("📚 Lecture Note Generator")
    st.write("Transform YouTube educational videos into comprehensive lecture notes")

    note_generator = LectureNoteGenerator()

    url = st.text_input("Enter YouTube Video URL:")

    if st.button("Generate Lecture Notes"):
        if url:
            with st.spinner("Processing video..."):
                video_id = note_generator.extract_video_id(url)
                if video_id:
                    video_info = note_generator.get_video_info(url)
                    if video_info:
                        st.header(video_info['title'])

                        transcript = note_generator.get_transcript(video_id)
                        if transcript:
                            notes = note_generator.generate_lecture_notes(transcript)

                            st.markdown("## 📝 Lecture Notes")

                            st.markdown("### 🎯 Introduction")
                            st.write(notes["Introduction"])

                            st.markdown("### 💡 Main Content")
                            for i, point in enumerate(notes["Main Content"], 1):
                                st.markdown(f"**Point {i}:**")
                                st.write(point)

                            st.markdown("### 🏁 Conclusion")
                            st.write(notes["Conclusion"])

                            notes_text = f"Introduction:\n{notes['Introduction']}\n\n"
                            notes_text += "Main Content:\n" + "\n".join(
                                [f"- {point}" for point in notes["Main Content"]]) + "\n\n"
                            notes_text += f"Conclusion:\n{notes['Conclusion']}"

                            st.download_button(
                                label="Download Notes",
                                data=notes_text,
                                file_name="lecture_notes.txt",
                                mime="text/plain"
                            )


if __name__ == "__main__":
    main()