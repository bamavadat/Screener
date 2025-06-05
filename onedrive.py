import requests
import re
import urllib.parse
import json
import time
import base64
from pathlib import Path
import tempfile
import os
from io import BytesIO
import zipfile
import xml.etree.ElementTree as ET
from urllib.parse import quote, unquote, urlparse, parse_qs
import hashlib


class OneDriveExtractor2025:
    def __init__(self):
        self.session = requests.Session()
        self.setup_session()

    def setup_session(self):
        """Setup session with realistic browser headers"""
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        })

    def get_onedrive_metadata(self, url):
        """Extract OneDrive file metadata from the sharing URL"""
        try:
            print("🔍 Analyzing OneDrive URL...")
            response = self.session.get(url, allow_redirects=True)

            if response.status_code != 200:
                print(f"❌ Failed to access URL: {response.status_code}")
                return None

            content = response.text
            final_url = response.url

            # Extract various metadata patterns
            metadata = {}

            # Pattern 1: Extract file information from script tags
            script_patterns = [
                r'window\["SdkModuleConfiguration"\]\s*=\s*({.*?});',
                r'window\.odsdkBoot\s*=\s*({.*?});',
                r'_spPageContextInfo\s*=\s*({.*?});',
                r'g_sharingLinkInfo\s*=\s*({.*?});'
            ]

            for pattern in script_patterns:
                matches = re.findall(pattern, content, re.DOTALL)
                if matches:
                    try:
                        data = json.loads(matches[0])
                        metadata.update(data)
                    except:
                        pass

            # Pattern 2: Extract direct download patterns
            download_patterns = [
                r'"downloadUrl":"([^"]+)"',
                r'"@microsoft\.graph\.downloadUrl":"([^"]+)"',
                r'"webUrl":"([^"]+)"',
                r'data-download-url="([^"]+)"',
                r'"redirectUrl":"([^"]+)"'
            ]

            download_urls = []
            for pattern in download_patterns:
                matches = re.findall(pattern, content)
                download_urls.extend(matches)

            # Pattern 3: Extract resource ID and other identifiers
            id_patterns = {
                'resource_id': r'"id":"([^"]+)"',
                'item_id': r'"itemId":"([^"]+)"',
                'drive_id': r'"driveId":"([^"]+)"',
                'site_id': r'"siteId":"([^"]+)"',
                'web_url': r'"webUrl":"([^"]+)"',
                'parent_reference': r'"parentReference":\s*({[^}]+})',
                'file_name': r'"name":"([^"]+\.docx?)"'
            }

            for key, pattern in id_patterns.items():
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    metadata[key] = matches[0]

            return {
                'metadata': metadata,
                'download_urls': download_urls,
                'final_url': final_url,
                'content': content
            }

        except Exception as e:
            print(f"❌ Error extracting metadata: {e}")
            return None

    def convert_to_direct_download(self, url):
        """Convert OneDrive sharing URL to direct download URL - 2025 method"""
        try:
            # Method 1: 1drv.ms URL transformation
            if '1drv.ms' in url:
                # Get the redirect first
                response = self.session.get(url, allow_redirects=True)
                actual_url = response.url
            else:
                actual_url = url

            # Method 2: URL parameter manipulation for 2025
            download_variations = []

            if 'sharepoint.com' in actual_url or 'onedrive.live.com' in actual_url:
                # Remove existing parameters and add download parameter
                base_url = actual_url.split('?')[0] if '?' in actual_url else actual_url

                # Extract query parameters
                parsed = urlparse(actual_url)
                params = parse_qs(parsed.query)

                # Create various download URL patterns that work in 2025
                download_variations = [
                    f"{base_url}?download=1",
                    f"{base_url}?e=download",
                    f"{actual_url}&download=1",
                    f"{actual_url}&action=download",
                    f"{actual_url}&format=download",
                    base_url.replace('/view.aspx', '/download.aspx'),
                    base_url.replace('/_layouts/15/Doc.aspx', '/_layouts/15/download.aspx'),
                    base_url + '?action=embedview&wdDownloadButton=True',
                    base_url + '?action=default&mobileredirect=true'
                ]

            return download_variations

        except Exception as e:
            print(f"❌ Error converting to direct download: {e}")
            return []

    def try_api_endpoints(self, metadata):
        """Try various API endpoints based on extracted metadata"""
        try:
            if not metadata or 'metadata' not in metadata:
                return None

            data = metadata['metadata']

            # Try Graph API endpoints
            if 'driveId' in data and 'itemId' in data:
                graph_url = f"https://graph.microsoft.com/v1.0/drives/{data['driveId']}/items/{data['itemId']}/content"
                try:
                    response = self.session.get(graph_url)
                    if response.status_code == 200:
                        return response.content
                except:
                    pass

            # Try SharePoint REST API
            if 'webUrl' in data:
                web_url = data['webUrl']
                if 'sharepoint.com' in web_url:
                    rest_endpoints = [
                        f"{web_url}/_api/web/getfilebyserverrelativeurl('{web_url}')/$value",
                        f"{web_url}/_api/web/lists/getbytitle('Documents')/items",
                    ]

                    for endpoint in rest_endpoints:
                        try:
                            headers = dict(self.session.headers)
                            headers['Accept'] = 'application/json;odata=verbose'
                            response = self.session.get(endpoint, headers=headers)
                            if response.status_code == 200:
                                return response.content
                        except:
                            continue

            return None

        except Exception as e:
            print(f"❌ Error trying API endpoints: {e}")
            return None

    def try_embed_download(self, url):
        """Try embed view with download capability"""
        try:
            # Convert to embed view
            if 'view.aspx' in url:
                embed_url = url.replace('view.aspx', 'embedview.aspx')
            else:
                separator = '&' if '?' in url else '?'
                embed_url = url + separator + 'action=embedview&wdDownloadButton=True&wdInConfigurator=True'

            print(f"🔗 Trying embed URL: {embed_url[:100]}...")

            # Get embed page
            response = self.session.get(embed_url)
            if response.status_code == 200:
                content = response.text

                # Look for download links in embed page
                download_patterns = [
                    r'href="([^"]*download[^"]*)"',
                    r'"downloadUrl":"([^"]+)"',
                    r'data-download-url="([^"]+)"',
                    r'"fileDownloadUrl":"([^"]+)"'
                ]

                for pattern in download_patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        download_url = match.replace('\\u0026', '&').replace('\\/', '/')
                        if not download_url.startswith('http'):
                            # Make absolute URL
                            if download_url.startswith('/'):
                                parsed = urlparse(embed_url)
                                download_url = f"{parsed.scheme}://{parsed.netloc}{download_url}"

                        try:
                            doc_response = self.session.get(download_url)
                            if doc_response.status_code == 200 and len(doc_response.content) > 1000:
                                return doc_response.content
                        except:
                            continue

            return None

        except Exception as e:
            print(f"❌ Error in embed download: {e}")
            return None

    def try_mobile_version(self, url):
        """Try mobile version which sometimes bypasses restrictions"""
        try:
            # Add mobile parameters
            mobile_params = [
                'mobileredirect=true',
                'mobile=1',
                'isMobileApp=1',
                'app=Word'
            ]

            for param in mobile_params:
                separator = '&' if '?' in url else '?'
                mobile_url = url + separator + param

                try:
                    headers = dict(self.session.headers)
                    headers['User-Agent'] = 'Microsoft Office Mobile'

                    response = self.session.get(mobile_url, headers=headers)
                    if response.status_code == 200 and len(response.content) > 1000:
                        return response.content
                except:
                    continue

            return None

        except Exception as e:
            print(f"❌ Error in mobile version: {e}")
            return None

    def extract_text_from_docx(self, docx_bytes):
        """Extract text from DOCX file bytes with enhanced error handling"""
        try:
            # Verify it's a valid ZIP file (DOCX format)
            if not docx_bytes.startswith(b'PK'):
                print("❌ Not a valid DOCX file (missing ZIP signature)")
                return None

            with zipfile.ZipFile(BytesIO(docx_bytes), 'r') as zip_file:
                # List files in the ZIP to debug
                file_list = zip_file.namelist()

                if 'word/document.xml' not in file_list:
                    print(f"❌ Invalid DOCX structure. Files found: {file_list[:10]}")
                    return None

                # Read the main document XML
                doc_xml = zip_file.read('word/document.xml')

                # Parse XML
                root = ET.fromstring(doc_xml)

                # Define namespace
                namespace = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

                # Extract text with proper paragraph separation
                paragraphs = root.findall('.//w:p', namespace)
                full_text = []

                for para in paragraphs:
                    para_text = ''
                    # Get all text runs in the paragraph
                    text_runs = para.findall('.//w:t', namespace)
                    for run in text_runs:
                        if run.text:
                            para_text += run.text

                    # Add paragraph if it has content
                    if para_text.strip():
                        full_text.append(para_text.strip())

                result = '\n\n'.join(full_text)

                if result.strip():
                    print(f"✅ Successfully extracted {len(result)} characters of text")
                    return result
                else:
                    print("❌ No text content found in document")
                    return None

        except zipfile.BadZipFile:
            print("❌ Invalid ZIP file format")
            return None
        except ET.ParseError as e:
            print(f"❌ XML parsing error: {e}")
            return None
        except Exception as e:
            print(f"❌ Error extracting text: {e}")
            return None

    def download_and_extract(self, url, output_file=None):
        """Main method to download and extract text - 2025 optimized"""
        print(f"🚀 Starting OneDrive document extraction...")
        print(f"📄 URL: {url}")

        # Step 1: Extract metadata
        metadata = self.get_onedrive_metadata(url)
        if not metadata:
            print("❌ Could not extract metadata")
            return None

        # Step 2: Try multiple download methods
        methods = [
            ("Direct URL conversion", lambda: self.try_direct_downloads(url)),
            ("Embed view download", lambda: self.try_embed_download(url)),
            ("Mobile version", lambda: self.try_mobile_version(url)),
            ("API endpoints", lambda: self.try_api_endpoints(metadata))
        ]

        for method_name, method_func in methods:
            print(f"🔄 Trying {method_name}...")

            try:
                content = method_func()
                if content and len(content) > 1000:
                    print(f"📦 Downloaded {len(content)} bytes with {method_name}")

                    # Extract text
                    text = self.extract_text_from_docx(content)
                    if text and text.strip():
                        print(f"✅ Success! Extracted text using {method_name}")

                        # Save to file
                        if output_file:
                            with open(output_file, 'w', encoding='utf-8') as f:
                                f.write(text)
                            print(f"💾 Text saved to: {output_file}")

                        return text
                    else:
                        print(f"❌ Could not extract readable text from content")
                else:
                    print(f"❌ No valid content found with {method_name}")

            except Exception as e:
                print(f"❌ {method_name} failed: {e}")
                continue

            # Small delay between attempts
            time.sleep(0.5)

        print("❌ All extraction methods failed")
        return None

    def try_direct_downloads(self, url):
        """Try all direct download URL variations"""
        download_urls = self.convert_to_direct_download(url)

        for download_url in download_urls:
            try:
                print(f"  🔗 Trying: {download_url[:80]}...")
                response = self.session.get(download_url, timeout=30)

                if response.status_code == 200 and len(response.content) > 1000:
                    # Check if it's actually a document
                    if response.content.startswith(b'PK'):  # ZIP/DOCX signature
                        return response.content

            except Exception as e:
                continue

        return None


# Utility function for batch processing
def process_multiple_urls(urls, output_dir="extracted_docs"):
    """Process multiple OneDrive URLs"""
    os.makedirs(output_dir, exist_ok=True)
    extractor = OneDriveExtractor2025()
    results = []

    for i, url in enumerate(urls):
        print(f"\n{'=' * 60}")
        print(f"Processing document {i + 1}/{len(urls)}")
        print(f"{'=' * 60}")

        output_file = os.path.join(output_dir, f"document_{i + 1}.txt")
        text = extractor.download_and_extract(url, output_file)

        results.append({
            'url': url,
            'success': bool(text),
            'text_length': len(text) if text else 0,
            'output_file': output_file if text else None
        })

        # Delay between documents to be respectful
        time.sleep(2)

    return results


# Main execution
if __name__ == "__main__":
    # Your URLs
    edit_url = "https://1drv.ms/w/c/08ba25ea884fc464/EfHzZoewvZFMp3mpeaf95egBd9PEW2umM21NJvuOFVoumw?e=dLsj1h"
    view_url = "https://1drv.ms/w/c/08ba25ea884fc464/EfHzZoewvZFMp3mpeaf95egB-dFT4FcqpIeu0KdYK1BH1A?e=70RGvL"

    # Create extractor
    extractor = OneDriveExtractor2025()

    # Try both URLs
    urls_to_try = [
        ("Edit URL", edit_url),
        ("View URL", view_url)
    ]

    for url_type, url in urls_to_try:
        print(f"\n{'=' * 70}")
        print(f"🔄 PROCESSING {url_type}")
        print(f"{'=' * 70}")

        output_file = f"extracted_{url_type.lower().replace(' ', '_')}.txt"
        text = extractor.download_and_extract(url, output_file)

        if text:
            print(f"\n{'=' * 50}")
            print(f"📄 EXTRACTED TEXT PREVIEW ({url_type}):")
            print(f"{'=' * 50}")
            preview = text[:1500] + "..." if len(text) > 1500 else text
            print(preview)
            print(f"\n📊 Total characters: {len(text)}")
            break  # Stop if we successfully extracted from one URL
        else:
            print(f"❌ Failed to extract from {url_type}")

    print(f"\n🏁 Extraction process completed!")
