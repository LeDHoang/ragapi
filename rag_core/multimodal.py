from typing import Dict, Any, List, Optional, Tuple
import logging
import base64
from pathlib import Path
from PIL import Image
import io
import json
import pandas as pd
from .config import config
from .llm_unified import UnifiedLLM
from .schemas import ContentType

logger = logging.getLogger(__name__)

class BaseModalProcessor:
    def __init__(self, llm: Optional[UnifiedLLM] = None):
        self.llm = llm or UnifiedLLM()
        self.config = config
    
    async def process_item(
        self,
        item: Dict[str, Any],
        context: str,
        doc_id: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Process a multimodal item"""
        raise NotImplementedError

class ImageProcessor(BaseModalProcessor):
    async def process_item(
        self,
        item: Dict[str, Any],
        context: str,
        doc_id: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Process image with context-aware analysis"""
        
        # Extract image metadata
        image_path = item["img_path"]
        captions = item.get("image_caption", [])
        footnotes = item.get("image_footnote", [])
        
        # Build analysis prompt
        prompt = f"""
        Analyze this image considering the surrounding context:
        
        Context: {context}
        Captions: {', '.join(captions) if captions else 'None'}
        Footnotes: {', '.join(footnotes) if footnotes else 'None'}
        
        Provide a comprehensive visual analysis including:
        1. Main subject/content
        2. Key visual elements
        3. Relationship to context
        4. Technical details (if relevant)
        5. Any text or annotations visible
        
        Format as JSON with these fields:
        - detailed_description: comprehensive visual analysis
        - key_elements: list of important elements
        - context_relevance: how it relates to surrounding content
        - technical_details: any technical aspects
        - extracted_text: any text visible in the image
        """
        
        try:
            # Get image data
            image_data = self._encode_image(image_path)
            
            # Analyze with vision model
            analysis = await self.llm.analyze_image(
                image_data=image_data,
                prompt=prompt,
                system_prompt="You are an expert image analyst specializing in document understanding."
            )
            
            # Parse analysis
            analysis_data = json.loads(analysis)
            
            # Create chunk content
            chunk_content = f"""
            Image Analysis:
            Path: {image_path}
            Page: {item.get('page_idx', 0)}
            
            Description:
            {analysis_data['detailed_description']}
            
            Key Elements:
            {', '.join(analysis_data['key_elements'])}
            
            Context Relevance:
            {analysis_data['context_relevance']}
            
            Technical Details:
            {analysis_data['technical_details']}
            
            Extracted Text:
            {analysis_data['extracted_text']}
            """
            
            # Create entity info
            entity_info = {
                "entity_name": f"image_{Path(image_path).stem}",
                "entity_type": "image",
                "description": analysis_data['detailed_description'][:200],
                "metadata": {
                    "key_elements": analysis_data['key_elements'],
                    "has_text": bool(analysis_data['extracted_text'].strip()),
                    "technical_details": analysis_data['technical_details']
                }
            }
            
            return chunk_content, entity_info
            
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            raise
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Convert to base64
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                return base64.b64encode(buffer.getvalue()).decode()
        except Exception as e:
            logger.error(f"Image encoding failed: {str(e)}")
            raise

class TableProcessor(BaseModalProcessor):
    async def process_item(
        self,
        item: Dict[str, Any],
        context: str,
        doc_id: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Process table with statistical analysis"""
        
        # Extract table data
        table_body = item["table_body"]
        captions = item.get("table_caption", [])
        
        try:
            # Convert markdown table to DataFrame
            df = self._parse_markdown_table(table_body)
            
            # Generate statistics
            stats = self._generate_table_stats(df)
            
            # Build analysis prompt
            prompt = f"""
            Analyze this table data considering context:
            
            Context: {context}
            Caption: {', '.join(captions) if captions else 'None'}
            
            Table Statistics:
            {json.dumps(stats, indent=2)}
            
            Table Data:
            {table_body}
            
            Provide a comprehensive analysis including:
            1. Key findings and patterns
            2. Statistical insights
            3. Relationship to context
            4. Notable data points
            5. Any trends or correlations
            
            Format as JSON with these fields:
            - main_findings: list of key findings
            - statistical_insights: key statistical observations
            - context_relevance: how it relates to surrounding content
            - notable_points: specific important data points
            - patterns: any identified patterns or trends
            """
            
            # Get analysis from LLM
            analysis = await self.llm.generate_text(
                prompt=prompt,
                system_prompt="You are an expert data analyst specializing in table understanding."
            )
            
            # Parse analysis
            analysis_data = json.loads(analysis)
            
            # Create chunk content
            chunk_content = f"""
            Table Analysis:
            Page: {item.get('page_idx', 0)}
            Caption: {', '.join(captions)}
            
            Main Findings:
            {', '.join(analysis_data['main_findings'])}
            
            Statistical Insights:
            {analysis_data['statistical_insights']}
            
            Context Relevance:
            {analysis_data['context_relevance']}
            
            Notable Points:
            {', '.join(analysis_data['notable_points'])}
            
            Patterns:
            {analysis_data['patterns']}
            
            Raw Data:
            {table_body}
            """
            
            # Create entity info
            entity_info = {
                "entity_name": f"table_{item.get('page_idx', 0)}_{hash(table_body)[:8]}",
                "entity_type": "table",
                "description": analysis_data['statistical_insights'][:200],
                "metadata": {
                    "dimensions": stats['dimensions'],
                    "column_types": stats['column_types'],
                    "has_numeric": stats['has_numeric']
                }
            }
            
            return chunk_content, entity_info
            
        except Exception as e:
            logger.error(f"Table processing failed: {str(e)}")
            raise
    
    def _parse_markdown_table(self, table_body: str) -> pd.DataFrame:
        """Parse markdown table to DataFrame"""
        try:
            # Split into lines and clean
            lines = [line.strip() for line in table_body.split('\n') if line.strip()]
            
            # Extract headers
            headers = [col.strip() for col in lines[0].split('|')[1:-1]]
            
            # Skip separator line
            data_lines = lines[2:]
            
            # Parse data
            data = []
            for line in data_lines:
                row = [cell.strip() for cell in line.split('|')[1:-1]]
                data.append(row)
            
            return pd.DataFrame(data, columns=headers)
        except Exception as e:
            logger.error(f"Table parsing failed: {str(e)}")
            raise
    
    def _generate_table_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate statistical information about table"""
        try:
            stats = {
                "dimensions": df.shape,
                "column_types": {},
                "has_numeric": False
            }
            
            for col in df.columns:
                # Try to convert to numeric
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if numeric_series.notna().any():
                    stats["has_numeric"] = True
                    stats["column_types"][col] = "numeric"
                else:
                    stats["column_types"][col] = "text"
            
            return stats
        except Exception as e:
            logger.error(f"Stats generation failed: {str(e)}")
            raise

class EquationProcessor(BaseModalProcessor):
    async def process_item(
        self,
        item: Dict[str, Any],
        context: str,
        doc_id: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Process equation with mathematical analysis"""
        
        latex = item["latex"]
        text = item.get("text", "")
        
        try:
            # Build analysis prompt
            prompt = f"""
            Analyze this mathematical equation in context:
            
            Context: {context}
            LaTeX: {latex}
            Description: {text}
            
            Provide a comprehensive analysis including:
            1. Mathematical meaning
            2. Components and variables
            3. Relationship to context
            4. Applications or implications
            5. Complexity assessment
            
            Format as JSON with these fields:
            - explanation: detailed explanation of the equation
            - components: list of key components and their meanings
            - context_relevance: how it relates to surrounding content
            - applications: potential applications or use cases
            - complexity_level: assessment of equation complexity (basic/intermediate/advanced)
            """
            
            # Get analysis from LLM
            analysis = await self.llm.generate_text(
                prompt=prompt,
                system_prompt="You are an expert mathematician specializing in equation understanding."
            )
            
            # Parse analysis
            analysis_data = json.loads(analysis)
            
            # Create chunk content
            chunk_content = f"""
            Equation Analysis:
            Page: {item.get('page_idx', 0)}
            LaTeX: {latex}
            
            Explanation:
            {analysis_data['explanation']}
            
            Components:
            {', '.join(analysis_data['components'])}
            
            Context Relevance:
            {analysis_data['context_relevance']}
            
            Applications:
            {analysis_data['applications']}
            
            Complexity Level:
            {analysis_data['complexity_level']}
            """
            
            # Create entity info
            entity_info = {
                "entity_name": f"equation_{item.get('page_idx', 0)}_{hash(latex)[:8]}",
                "entity_type": "equation",
                "description": analysis_data['explanation'][:200],
                "metadata": {
                    "complexity": analysis_data['complexity_level'],
                    "components": analysis_data['components'],
                    "latex": latex
                }
            }
            
            return chunk_content, entity_info
            
        except Exception as e:
            logger.error(f"Equation processing failed: {str(e)}")
            raise

class MultimodalProcessor:
    def __init__(self):
        self.llm = UnifiedLLM()
        self.processors = {
            ContentType.IMAGE: ImageProcessor(self.llm),
            ContentType.TABLE: TableProcessor(self.llm),
            ContentType.EQUATION: EquationProcessor(self.llm)
        }
    
    async def process_item(
        self,
        item: Dict[str, Any],
        context: str,
        doc_id: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Process a multimodal item with appropriate processor"""
        
        content_type = item.get("type")
        processor = self.processors.get(content_type)
        
        if not processor:
            raise ValueError(f"No processor available for content type: {content_type}")
        
        return await processor.process_item(item, context, doc_id)
