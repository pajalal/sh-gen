function mul(a, b)
{
    c = [];
    for (let x of a) 
    for (let y of b)
        c.push([x[0] * y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3]]);
    return c;
}

function process_adds(a)
{
    let m = {};  
    let g = [];
    let z = [];  
    for (let x of a)
    {
        let q = x.slice(1); 
        if (q in m) z[m[q]] += x[0];
        else
        {
            m[q] = z.length
            z.push(x[0]);
            g.push(q);
        }
    }
    let l = [];
    for (let i = 0; i < g.length; ++i)
        l.push([z[i], ...g[i]]);
    return l;
}

function add(a, b)
{
    return process_adds([].concat(a, b));
}

function number(value)
{
    return [[value, 0, 0, 0]];
}

function label(num)
{
    let v = [1, 0, 0, 0];
    v[1 + num] = 1;
    return [v];
}

function sub(a, b)
{
    return add(a, mul(number(-1), b));
}

function S(m)
{
    if (m == 0) return number(0);
	let a = mul(label(0), S(m - 1));
	let b = mul(label(2), C(m - 1));
	return add(a, b);
}

function C(m)
{
    if (m == 0) return number(1);
	let a = mul(label(0), C(m - 1));
	let b = mul(label(2), S(m - 1));
	return sub(a, b);
}

function P(m, l)
{
    if (m == 0 && l == 0) return number(1);
	else if (m == l) return mul(number(1 - 2*m), P(m - 1, l - 1));
	else if (l == m + 1) return mul(P(m, l - 1), mul(number(2*m + 1), label(1)));
    let X = mul(label(1), number((2*l - 1) / (l - m)));
    let A = mul(X, P(m, l - 1));
    let B = mul(number((1 - l - m) / (l - m)), P(m, l - 2));
    return add(A, B);
}

let factorial_table = [1];

for (let i = 1; i < 32; ++i)
    factorial_table.push(i * factorial_table[i - 1]);

function factorial(k)
{
    if (k < 32) return factorial_table[k];
    else return Math.sqrt(2 * Math.PI * k) * Math.pow(k / Math.E, k);
}

function K(m, l)
{
    let div = 4.0 * Math.PI * factorial(l + Math.abs(m));
    return Math.sqrt((2*l + 1) * factorial(l - Math.abs(m)) / div);
}

function Y_aux(m, l)
{
    const sqrt2 = Math.sqrt(2);
    let k = K(m, l);
	if      (m > 0) return mul(number(sqrt2 * k), mul(C( m), P( m, l)));
	else if (m < 0) return mul(number(sqrt2 * k), mul(S(-m), P(-m, l)));
	return process_adds(mul(number(k), P(m, l)));
}

function power(label, exponent)
{
    let v = "xyz"[label];
    if (exponent > 1)
    {
        if (exponent < 4)
            return (v + "*").repeat(exponent - 1) + v;
        return "Math.pow(" + v + ", " + exponent + ")";
    }
    return v;
}

function make_lets(xs)
{
    let defs = [];  
    for (let i = 0; i < 3; ++i)
    {
        let v = "xyz"[i];
        let pows = [...xs[i]].sort();
        for (let j = 0; j < pows.length; ++j)
        {
            let left = pows[j];
            if (left == 1) continue;
            let factors = [];       
            for (let k = j - 1; k >= 0; --k)
            {
                if (pows[k] != 1 && (left - pows[k]) > 0)
                {
                    factors.push(v + pows[k]);
                    left -= pows[k];
                }
            }
            for (let k = 0; k < left; ++k)
                factors.push(v);
            defs.push(v + pows[j] + "=" + factors.join("*"));
        }
    }
    return defs;
}

function Y_to_func(y)
{
    let factors = [];
    let xs = [new Set(), new Set(), new Set()];
    for (let f of y)
    {
        if (f[0] == 0) continue;
        let terms = [f[0]];
        for (let i = 0; i < 3; ++i)
        {
            if (f[1 + i] == 0) continue;
            if (f[1 + i] > 1) terms.push("xyz"[i] + f[1 + i]);
            else              terms.push("xyz"[i]);
            xs[i].add(f[1 + i]);
        }
        factors.push(terms.join("*"));
    }
    let body = "return " + factors.join(" + ").replaceAll(" + -", " - ").replaceAll("* ", "");
    let lets = make_lets(xs);
    if (lets.length)
        return eval("(x,y,z) => { let " + lets.join(",") + "; " + body + "; };");
    return eval("(x,y,z) => { " + body + "; };");
}

function Y_to_text(y)
{
    let factors = [];
    for (let f of y)
    {
        if (f[0] == 0) continue;
        let terms = [f[0]];
        for (let i = 0; i < 3; ++i)
        {
            if (f[1 + i] == 0) continue;
            if (f[1 + i] > 1) terms.push("xyz"[i] + "^" + f[1 + i]);
            else              terms.push("xyz"[i]);
        }
        factors.push(terms.join("*"));
    }
    let text = factors.join(" + ").replaceAll(" + -", " - ").replaceAll("* ", "");
    return text.replaceAll(" + ", " +\n").replaceAll(" - ", " -\n");
}

function Y(m, l)
{
    let y = Y_aux(m, l);
    return [Y_to_func(y), Y_to_text(y)];
}

function color_grad(x)
{
    let r = x > 0 ? x : 0;
    let g = x > 0 ? 1 - x : 1 + x;
    let b = x < 0 ? -x : 0;
    return [r, g, b];
}

function gen_radial_mesh(func, num)
{
    let vertices = [];
    let rs = [];
    let max_r = 0;
    let min_r = 10e10;
    function push_vertex(x, y, z)
    {
        let r = func(x, y, z)
        vertices.push([1, 0, 0]);
        rs.push(r);
        r = Math.abs(r);
        max_r = Math.max(max_r, r);
        min_r = Math.min(min_r, r);
        if (rendermode == 1) r = 0.5;
        vertices.push([r*x, r*y, r*z]);
        vertices.push([0, 0, 0]);
    }
    push_vertex(0, -1, 0);   
    const factor = Math.PI / num;
    for (let a = 1 - num/2; a < num/2; a++)
    for (let b = 0; b < num; b++)
    {
        let u = a * factor;
        let v = 2 * b * factor;
        let x = Math.cos(v) * Math.cos(u);
        let y = Math.sin(u);
        let z = Math.sin(v) * Math.cos(u);
        push_vertex(x, y, z);
    }
    push_vertex(0, 1, 0);
    if (max_r != min_r)
    {
        let scale = 1 / (max_r - min_r);
        for (let i = 0; i < vertices.length / 3; ++i)
        {
            let x = rs[i];
            let y = (Math.abs(rs[i]) - min_r) * scale;
            vertices[3*i][0] = x > 0 ? y : 0;
            vertices[3*i][1] = 1 - y;
            vertices[3*i][2] = x < 0 ? y : 0;
        }
    }
    let tris = [];
    for (let a = 0; a < num; a++)
        tris.push(0, 1 + a, 1 + (1 + a) % num);
    for(let a = 1; a < num - 2; a++)
    for(let b = 0; b < num; b++)
    {
        let bn = (b + 1) % num;
        let v0 = 1 + a*num + b;
        let v1 = 1 + a*num + bn;
        let v2 = 1 + (a + 1)*num + b;
        let v3 = 1 + (a + 1)*num + bn;
        tris.push(v0, v2, v1);
        tris.push(v2, v3, v1);
    }
    let l = vertices.length / 3 - 1;
    for(let a = 0; a < num; a++)
    {
        let k = 1 + num*(num - 2);
        tris.push(l, k, k + (a % num));
    }
    return [vertices, tris];
}

function projection(fov, aspect, near, far)
{
    const a = 1.0 / Math.tan(0.5 * fov);
    const b = a / aspect;
    const c = -((far + near) / (far - near));
    const d = -((2.0 * far * near) / (far - near));
    return [
        [ b,  0,  0,  0 ],
        [ 0,  a,  0,  0 ],
        [ 0,  0,  c, -1 ],
        [ 0,  0,  d,  0 ]
    ];
}

function identity()
{
    return [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ];
}

function vdot(a, b)
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

function vadd(r, a, b)
{
    r[0] = a[0] + b[0];
    r[1] = a[1] + b[1];
    r[2] = a[2] + b[2];
}

function vsub(r, a, b)
{
    r[0] = a[0] - b[0];
    r[1] = a[1] - b[1];
    r[2] = a[2] - b[2];
}

function vcross(r, a, b)
{
    let x = a[1] * b[2] - a[2] * b[1];
    let y = a[2] * b[0] - a[0] * b[2];
    let z = a[0] * b[1] - a[1] * b[0];
    r[0] = x;
    r[1] = y;
    r[2] = z;
}

function vlength(v)
{
    return Math.sqrt(vdot(v, v));
}

function vnormalize(r, v)
{
    const f = 1.0 / vlength(v);
    r[0] = f * v[0];
    r[1] = f * v[1];
    r[2] = f * v[2];
}

function lookAt(eye, pos, up)
{
    let x = [0, 0, 0];
    let y = [0, 0, 0];
    let z = [0, 0, 0]; 
    vsub(z, pos, eye);
    vnormalize(z, z);  
    vcross(x, z, up);
    vnormalize(x, x);
    vcross(y, x, z);
    return [
        [x[0], y[0], -z[0], 0],
        [x[1], y[1], -z[1], 0],
        [x[2], y[2], -z[2], 0],
        [0, 0, eye[2], 1]
    ];
}

function vertical_rotate(angle)
{
    let s = Math.sin(angle);
	let c = Math.cos(angle);
	return [
		[ c, 0,-s, 0 ],
		[ 0, 1, 0, 0 ],
		[ s, 0, c, 0 ],
		[ 0, 0, 0, 1 ]
	];
}

function compile_shader(gl, type, source)
{
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    return shader;
}

function create_program(gl, shaders)
{
    const prog = gl.createProgram();
    for (let shader of shaders)
        gl.attachShader(prog, shader);
    gl.linkProgram(prog);
    return prog;
}

function main() 
{
    const canvas = document.getElementById("thecanvas");
    
    let width  = document.body.clientWidth;
    let height = document.body.clientHeight;
    
    canvas.width  = width;
    canvas.height = height;
    
    const poly = document.getElementById("poly");
    const gl = canvas.getContext("webgl");

    if (gl === null) 
    {
        alert("WebGL had an oopsie!");
        return;
    }

    const inds_buf  = gl.createBuffer();
    const verts_buf = gl.createBuffer();
    let num_elems = 0;
    
    function resize()
    {
        width  = document.body.clientWidth;
        height = document.body.clientHeight;
    }
    
    window.addEventListener('resize', resize);
    
    function calculate(m, l)
    {
        let y = Y(m, l);
        poly.innerHTML = "Y(" + m + ", " + l + ") = \n" + y[1];
        let mesh  = gen_radial_mesh(y[0], 128); 
        let verts = mesh[0];
        let tris  = mesh[1];
        let num_verts = verts.length / 3;
        let tri_normals = [];
        let vert_tris   = [];
        for (let i = 0; i < num_verts; ++i)
            vert_tris.push([]);
        num_elems = tris.length;
        for (let i = 0; i < tris.length / 3; ++i)
        {
            let v0 = verts[3*tris[3*i + 0] + 1];
            let v1 = verts[3*tris[3*i + 1] + 1];
            let v2 = verts[3*tris[3*i + 2] + 1];
            let e0 = [0, 0, 0];
            let e1 = [0, 0, 0];
            let nm = [0, 0, 0];
            vsub(e0, v1, v0);
            vsub(e1, v2, v0);
            vcross(nm, e0, e1);
            vnormalize(nm, nm);
            tri_normals.push(nm);
            for (let j = 0; j < 3; ++j)
                vert_tris[tris[3*i + j]].push(i);
        }    
        for (let i = 0; i < num_verts; ++i)
        {
            let nm = [0, 0, 0];
            for (let j of vert_tris[i])
                vadd(nm, nm, tri_normals[j]);
            vnormalize(nm, nm);
            verts[3*i + 2] = nm;
        }       
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, inds_buf);
        gl.bufferData
        (
            gl.ELEMENT_ARRAY_BUFFER,
            new Uint16Array(tris),
            gl.STATIC_DRAW
        );
        gl.bindBuffer(gl.ARRAY_BUFFER, verts_buf);
        gl.bufferData
        (
            gl.ARRAY_BUFFER,
            new Float32Array(verts.flat()),
            gl.STATIC_DRAW
        );
    }
    
    let vshader_source = `
        attribute vec3 position;
        attribute vec3 normal;
        attribute vec3 color;
        uniform mat4 view;
        uniform mat4 proj;
        uniform mat4 model;
        varying highp vec3 pos;
        varying highp vec3 norm;
        varying highp vec3 col;
        void main() 
        {
            vec4 p = model * vec4(position, 1);
            gl_Position  = proj * view * p;
            gl_PointSize = 3.0;
            pos = p.xyz;
            norm = (model * vec4(normal, 0)).xyz;
            col = color;
        }
    `;
    
    let fshader_source = `
        varying highp vec3 pos;
        varying highp vec3 norm;
        varying highp vec3 col;
        void main() 
        {
            highp vec3 L = normalize(vec3(4, 4, -4) - pos);
            highp vec3 N = norm;
            highp vec3 V = normalize(pos - vec3(0, 0, -1));
            highp vec3 R = reflect(L, N);
            if (dot(V, N) > 0.0) N = -N;
            highp float diffuse  = max(0.0, dot(L, N));
            highp float specular = max(0.0, dot(V, R));
            specular = pow(specular, 32.0);
            gl_FragColor = vec4(col*vec3(0.3 + 0.7*diffuse + 0.3*specular), 1.0);
        }
    `;
    
    let pickL = document.getElementById("selectL");
    let pickM = document.getElementById("selectM");
    
    let curL = +pickL.value;
    let curM = +pickM.value;
    
    calculate(curM, curL);
    
    let vshader = compile_shader(gl, gl.VERTEX_SHADER, vshader_source);
    let fshader = compile_shader(gl, gl.FRAGMENT_SHADER, fshader_source);  
    let prog = create_program(gl, [vshader, fshader]);      
    gl.useProgram(prog);   
    let vertex_pos = gl.getAttribLocation (prog, "position");
    let normal_pos = gl.getAttribLocation (prog, "normal");
    let color_pos  = gl.getAttribLocation (prog, "color");
    let proj_loc   = gl.getUniformLocation(prog, "proj");
    let view_loc   = gl.getUniformLocation(prog, "view");
    let modl_loc   = gl.getUniformLocation(prog, "model");   
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.clearDepth(1.0);
    gl.clearColor(0, 0, 0, 1);  
    
    let rmod = 0;
    
    function render(now)
    {
        let L = +pickL.value;
        let M = +pickM.value;      
        let R = rendermode;
        if (L != curL || M != curM || R != rmod)
        {
            if (Math.abs(M) <= L || rmod)
            {
                calculate(M, L);
                curL = L;
                curM = M;
                rmod = R;
            }   
        }
        let proj = projection(0.4*Math.PI, width / height, 0.01, 10.0);
        let view = lookAt([0, 0, -1.3], [0, 0, 0], [0, 1 ,0]);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);      
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, inds_buf);                 
        gl.bindBuffer(gl.ARRAY_BUFFER, verts_buf);    
        gl.vertexAttribPointer( color_pos, 3, gl.FLOAT, false, 36,  0);
        gl.vertexAttribPointer(vertex_pos, 3, gl.FLOAT, false, 36, 12);
        gl.vertexAttribPointer(normal_pos, 3, gl.FLOAT, false, 36, 24);
        gl.enableVertexAttribArray(vertex_pos);
        gl.enableVertexAttribArray(normal_pos);
        gl.enableVertexAttribArray(color_pos);
        let vrot = vertical_rotate(0.001*now);
        if (paused) vrot = identity();
        gl.useProgram(prog); 
        gl.uniformMatrix4fv(proj_loc, false, proj.flat());
        gl.uniformMatrix4fv(view_loc, false, view.flat());
        gl.uniformMatrix4fv(modl_loc, false, vrot.flat());
        gl.viewport(0, 0, width, height);
        gl.drawElements(gl.TRIANGLES, num_elems, gl.UNSIGNED_SHORT, 0);     
        requestAnimationFrame(render);
    }  
    requestAnimationFrame(render);
}

main();