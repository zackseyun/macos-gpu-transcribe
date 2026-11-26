import json
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from unittest.mock import patch
import nemotron_resident as svc


class ResidentTest(unittest.TestCase):
    def test_new_clients_share_one_ready_service(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as d, socket.socket(socket.AF_UNIX) as server:
            path = Path(d) / "asr.sock"
            server.bind(str(path))
            server.listen()
            def serve():
                for _ in range(2):
                    conn, _ = server.accept()
                    with conn, conn.makefile("rb") as stream:
                        data = json.loads(stream.readline())
                        conn.sendall(json.dumps({"ready": data["warm"], "pid": 12}).encode() + b"\n")
            t = threading.Thread(target=serve, daemon=True)
            t.start()
            with patch.object(svc, "SOCKET", path):
                for _ in range(2):
                    self.assertEqual(svc.request({"warm": True}), {"ready": True, "pid": 12})
            t.join(1)
            self.assertFalse(t.is_alive())

    def test_missing_service_does_not_launch_a_second_model(self):
        with tempfile.TemporaryDirectory() as d, patch.object(svc, "SOCKET", Path(d) / "absent"):
            with self.assertRaises(TimeoutError):
                svc.request({"warm": True}, wait=0)
