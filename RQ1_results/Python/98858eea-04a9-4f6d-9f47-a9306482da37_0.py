def bulk_ingest(self, s3_paths: Union[List[str], str], course_name: str, **kwargs) -> Dict[str, List[str]]:
    success_status = {"success_ingest": [], "failure_ingest": []}

    def ingest(file_ext_mapping, s3_path, *args, **kwargs):
        handler = file_ext_mapping.get(Path(s3_path).suffix)
        if handler:
            ret = handler(s3_path, *args, **kwargs)
            if ret != "Success":
                success_status['failure_ingest'].append(s3_path)
            else:
                success_status['success_ingest'].append(s3_path)

    file_ext_mapping = {
        '.html': self._ingest_html,
        '.py': self._ingest_single_py,
        '.vtt': self._ingest_single_vtt,
        '.pdf': self._ingest_single_pdf,
        '.txt': self._ingest_single_txt,
        '.md': self._ingest_single_txt,
        '.srt': self._ingest_single_srt,
        '.docx': self._ingest_single_docx,
        '.ppt': self._ingest_single_ppt,
        '.pptx': self._ingest_single_ppt,
    }

    try:
        if isinstance(s3_paths, str):
            s3_paths = [s3_paths]

        for s3_path in s3_paths:
            with NamedTemporaryFile(suffix=Path(s3_path).suffix) as tmpfile:
                self.s3_client.download_fileobj(Bucket=os.environ['S3_BUCKET_NAME'], Key=s3_path, Fileobj=tmpfile)
                mime_type = mimetypes.guess_type(tmpfile.name)[0]
                category, _ = mime_type.split('/')

            if category in ['video', 'audio']:
                ret = self._ingest_single_video(s3_path, course_name)
                if ret != "Success":
                    success_status['failure_ingest'].append(s3_path)
                else:
                    success_status['success_ingest'].append(s3_path)
            else:
                ingest(file_ext_mapping, s3_path, course_name, kwargs=kwargs)

        return success_status
    except Exception as e:
        success_status['failure_ingest'].append(f"MAJOR ERROR IN /bulk_ingest: Error: {str(e)}")
        return success_status
