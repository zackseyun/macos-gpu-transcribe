import base64
import json
import types
import unittest
from unittest.mock import Mock, patch
import numpy as np
import mlx_next_asr as backend


class NextASRTest(unittest.TestCase):
    def test_unknown_mode_fails_before_launch(self):
        with self.assertRaises(ValueError):
            backend.NextASR().warm('invalid')

    def test_warm_resident_model_does_not_reload(self):
        client = backend.NextASR()
        client.mode = 'granite5'
        client.process = Mock()
        client.process.poll.return_value = None
        with patch.object(client, '_request') as request:
            client.warm('granite5')
            request.assert_not_called()

    def test_timeout_stops_child(self):
        client = backend.NextASR()
        client.process = Mock()
        with patch('select.select', return_value=([], [], [])), patch.object(client, 'stop') as stop:
            with self.assertRaises(TimeoutError):
                client._request({'warm': True}, timeout=0)
            stop.assert_called_once()

    def test_server_error_stops_child(self):
        client = backend.NextASR()
        client.process = Mock()
        client.process.stdout.readline.return_value = json.dumps({'error': 'load failed'})+'\n'
        with patch('select.select', return_value=([True], [], [])), patch.object(client, 'stop') as stop:
            with self.assertRaisesRegex(RuntimeError, 'load failed'):
                client._request({'warm': True})
            stop.assert_called_once()

    def test_transcribe_uses_private_pipe_wav(self):
        client = backend.NextASR()
        with patch.object(client, 'warm') as warm, patch.object(client, '_request', return_value={'text': 'Hello.'}) as request:
            self.assertEqual(client.transcribe((np.zeros(1600), 16000), 'granite5'), 'Hello.')
            warm.assert_called_once_with('granite5')
            self.assertTrue(base64.b64decode(request.call_args.args[0]['wav']).startswith(b'RIFF'))

    def test_granite_preserves_language_and_requests_punctuation(self):
        model = Mock()
        model.generate.return_value = types.SimpleNamespace(text='Hello.', generation_tokens=2)
        self.assertEqual(backend.generate(model, np.zeros(16000), 'granite5'), 'Hello.')
        kwargs = model.generate.call_args.kwargs
        self.assertTrue(kwargs['punctuate'])
        self.assertNotIn('language', kwargs)  # language would translate rather than transcribe
        self.assertEqual(kwargs['temperature'], 0)

    def test_truncation_is_an_error(self):
        model = Mock()
        for output in [types.SimpleNamespace(text='Partial', finish_reason='length'),
                       types.SimpleNamespace(text='Partial', generation_tokens=128)]:
            model.generate.return_value = output
            with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                backend.generate(model, np.zeros(16000), 'granite5')

    def test_long_audio_is_chunked_and_overlap_deduplicated(self):
        model = Mock()
        model.generate.side_effect = [types.SimpleNamespace(text='One two three.'),
                                      types.SimpleNamespace(text='Two three, Four.')]
        self.assertEqual(backend.generate(model, np.zeros(115*16000), 'granite5'), 'One two three. Four.')
        self.assertEqual(model.generate.call_count, 2)
        self.assertEqual([len(c.args[0]) for c in model.generate.call_args_list], [110*16000, 7*16000])

if __name__ == '__main__':
    unittest.main()
