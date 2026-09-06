import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import transcribe

class PasteDeliveryTest(unittest.TestCase):
    def test_delayed_restore_respects_new_clipboard_write_even_with_same_text(self):
        for user_copied in (False, True):
            state = {'text': 'old clipboard', 'count': 1}
            pb = Mock()
            pb.stringForType_.side_effect = lambda _: state['text']
            pb.changeCount.side_effect = lambda: state['count']
            def clear():
                state['text'] = None
                state['count'] += 1
            def write(text, _):
                state['text'] = text
                state['count'] += 1
                return True
            pb.clearContents.side_effect = clear
            pb.setString_forType_.side_effect = write
            app = SimpleNamespace(_paste_lock=threading.Lock(), _paste_generation=0)
            with patch('AppKit.NSPasteboard', SimpleNamespace(generalPasteboard=lambda: pb)), \
                 patch('Quartz.CGEventCreateKeyboardEvent'), patch('Quartz.CGEventSetFlags'), \
                 patch('Quartz.CGEventSourceCreate'), patch('Quartz.CGEventPost') as post, \
                 patch('transcribe.time.sleep'), patch('transcribe.threading.Thread') as thread:
                self.assertTrue(transcribe.VoiceTranscribeApp._paste_text(app, 'new transcript'))
                self.assertEqual(post.call_count, 2)
                self.assertEqual(state['text'], 'new transcript')
                if user_copied:
                    write('new transcript', None)
                thread.call_args.kwargs['target']()
                self.assertEqual(state['text'], 'new transcript' if user_copied else 'old clipboard')
