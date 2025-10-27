#!/usr/bin/env python3
"""
Integration test for the enhanced multimodal RAG system
"""
import asyncio
import os
from pathlib import Path
from rag_core.config import AppConfig
from rag_core.parsers import make_parser
from rag_core.processors import ModalProcessors
from rag_core.llm_unified import create_llm
from rag_core.context_extractor import ContextExtractor
from rag_core.overlay import LayoutOverlay

async def test_multimodal_processing():
    """Test the complete multimodal processing pipeline"""

    # Initialize configuration
    cfg = AppConfig.from_env()
    print(f"Using parser: {cfg.parser}")
    print(f"LLM provider: {cfg.llm_provider}")
    print(f"Layout overlay: {cfg.export_layout_overlay}")

    # Test parser
    parser = make_parser(cfg)
    print(f"Parser created: {type(parser)}")

    # Test LLM
    llm = create_llm(cfg)
    print(f"LLM created: {type(llm)}")

    # Test processors with context
    processors = ModalProcessors.from_config(cfg, llm=llm)
    print(f"Processors created with context extractor: {processors.context_extractor is not None}")

    # Test overlay generator
    overlay_gen = LayoutOverlay(cfg) if cfg.export_layout_overlay else None
    print(f"Overlay generator: {overlay_gen is not None}")

    # Test with sample file if available
    input_dir = Path("input")
    if input_dir.exists():
        pdf_files = list(input_dir.glob("*.pdf"))
        if pdf_files:
            test_pdf = pdf_files[0]
            print(f"Testing with: {test_pdf}")

            # Parse content
            content_list = parser.parse_pdf(test_pdf)
            print(f"Parsed {len(content_list)} content items")

            # Analyze content types
            by_type = {}
            for item in content_list:
                t = item.get("type", "unknown")
                by_type[t] = by_type.get(t, 0) + 1

                # Check for enhanced metadata
                has_bbox = item.get("bbox") is not None
                has_page_size = item.get("page_size") is not None
                print(f"  {t}: bbox={has_bbox}, page_size={has_page_size}")

            print(f"Content breakdown: {by_type}")

            # Test context extraction
            if processors.context_extractor:
                sample_item = content_list[0] if content_list else {}
                context = processors.context_extractor.extract_context(content_list, sample_item)
                print(f"Context extracted: {len(context)} characters")

            # Test LLM descriptions
            if content_list and llm:
                for item in content_list[:3]:  # Test first 3 items
                    try:
                        desc = await processors.describe_item(item)
                        print(f"Description for {item.get('type')}: {desc[:100]}...")
                    except Exception as e:
                        print(f"Error describing item: {e}")

            # Test layout overlay if enabled
            if overlay_gen and cfg.export_layout_overlay:
                try:
                    result = overlay_gen.render(test_pdf, content_list)
                    print(f"Overlay generated: {result}")
                except Exception as e:
                    print(f"Overlay generation failed: {e}")
        else:
            print("No PDF files found in input directory")
    else:
        print("Input directory not found")

    print("Multimodal processing test completed!")

if __name__ == "__main__":
    asyncio.run(test_multimodal_processing())
