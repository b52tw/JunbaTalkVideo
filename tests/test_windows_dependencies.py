"""Audit the downloaded Windows wheels, including Windows-only dependencies.
Set JUNBA_TEST_WINDOWS_WHEELS to the directory used by the release validation.
"""
import email,hashlib,json,os,unittest,zipfile
from pathlib import Path

class WindowsDependencies(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('JUNBA_TEST_WINDOWS_WHEELS'),'release wheel cache not supplied')
    def test_complete_pinned_windows_dependency_closure(self):
        from packaging.requirements import Requirement
        from packaging.utils import canonicalize_name
        from packaging.markers import default_environment
        folder=Path(os.environ['JUNBA_TEST_WINDOWS_WHEELS'])
        rows=json.loads((Path(__file__).resolve().parents[1]/'local/sadtalker_wheels.json').read_text())
        packages={};requirements=[]
        for row in rows:
            p=folder/row['name'];h=hashlib.sha256()
            with p.open('rb') as f:
                for block in iter(lambda:f.read(1048576),b''):h.update(block)
            self.assertEqual(h.hexdigest(),row['sha256'],p.name)
            with zipfile.ZipFile(p) as z:
                meta=email.message_from_bytes(z.read(next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))))
            packages[canonicalize_name(meta['Name'])]=meta['Version']
            requirements.extend((meta['Name'],r) for r in meta.get_all('Requires-Dist',[]))
        env=default_environment();env.update(sys_platform='win32',platform_system='Windows',os_name='nt',python_version='3.11',python_full_version='3.11.9',platform_machine='AMD64',extra='')
        for parent,spec in requirements:
            r=Requirement(spec)
            if r.marker and not r.marker.evaluate(env):continue
            name=canonicalize_name(r.name)
            self.assertIn(name,packages,f'{parent}: {r}')
            self.assertTrue(not r.specifier or packages[name] in r.specifier,f'{parent}: {r}, got {packages[name]}')

if __name__=='__main__':unittest.main()
