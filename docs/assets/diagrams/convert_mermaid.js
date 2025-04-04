const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Install mermaid-cli if not already installed
try {
  execSync('npx -v', { stdio: 'ignore' });
  console.log('npx is available, proceeding with conversion...');
} catch (error) {
  console.error('npx is not available. Please install Node.js and npm.');
  process.exit(1);
}

const diagramsDir = __dirname;
const files = fs.readdirSync(diagramsDir).filter(file => file.endsWith('.md'));

files.forEach(file => {
  const filePath = path.join(diagramsDir, file);
  const content = fs.readFileSync(filePath, 'utf8');
  
  // Extract the mermaid code
  const mermaidMatch = content.match(/```mermaid\n([\s\S]*?)```/);
  if (!mermaidMatch) {
    console.error(`No mermaid code found in ${file}`);
    return;
  }
  
  const mermaidCode = mermaidMatch[1];
  const tempFile = path.join(diagramsDir, `${path.basename(file, '.md')}.mmd`);
  const outputFile = path.join(diagramsDir, `${path.basename(file, '.md')}.svg`);
  
  // Write the mermaid code to a temporary file
  fs.writeFileSync(tempFile, mermaidCode);
  
  try {
    // Convert the mermaid file to SVG using mermaid-cli
    execSync(`npx @mermaid-js/mermaid-cli -i ${tempFile} -o ${outputFile}`, { stdio: 'inherit' });
    console.log(`Converted ${file} to SVG`);
  } catch (error) {
    console.error(`Error converting ${file} to SVG:`, error.message);
  }
  
  // Clean up the temporary file
  fs.unlinkSync(tempFile);
});

console.log('Conversion complete!');
